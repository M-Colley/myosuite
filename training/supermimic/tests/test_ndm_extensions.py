# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for NDM Gym extensions: acceleration world model, RigidFormer backbone, SuperTrack, smooth targets, and inference copies."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")


def _build_hinge_spec(self_collisions: bool = False) -> Any:
    """Build a ``myofullbody`` spec compatible with ``SkeletonKinematics`` across ``myo_sim`` 0.2 and 0.3."""
    import mujoco
    import numpy as np

    from myosuite.envs.myo.tasks.mimic.tracking.models import (
        INIT_POS,
        INIT_ROT,
        ROOT_BODY,
        _myofullbody_spec,
    )

    try:
        spec = _myofullbody_spec(self_collisions)
    except KeyError:
        import myo_sim
        from myo_sim.build.hand import weld_hand_fingers

        from myosuite.integrations.musclemimic.fullbody_model import (
            FULLBODY_BODY2SITES_FOR_MIMIC,
            _apply_mimic_fullbody_spec_changes,
        )

        spec = myo_sim.load_spec("myofullbody")
        weld_hand_fingers(spec)
        spec = _apply_mimic_fullbody_spec_changes(spec, False, (), ())
        spec.njmax = spec.nconmax = spec.nuser_jnt = -1
        spec.option.timestep = mujoco.MjSpec().option.timestep
        for key in list(spec.keys):
            spec.delete(key)
        for geom in list(spec.worldbody.geoms):
            spec.delete(geom)
        for light in list(spec.worldbody.lights):
            spec.delete(light)
        root = spec.body(ROOT_BODY)
        root.pos, root.quat = INIT_POS, INIT_ROT
        for site in list(spec.sites):
            if site.name in FULLBODY_BODY2SITES_FOR_MIMIC.values():
                spec.delete(site)
        for sensor in list(spec.sensors):
            if sensor.type != mujoco.mjtSensor.mjSENS_TOUCH:
                spec.delete(sensor)
        if not self_collisions:
            for pair in list(spec.pairs):
                spec.delete(pair)

    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_BALL:
            j.type = mujoco.mjtJoint.mjJNT_HINGE
            if float(np.linalg.norm(j.axis)) < 1e-6:
                j.axis = (1.0, 0.0, 0.0)
            if not j.limited:
                j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
                j.range = (-0.5, 0.5)

    # Filter any myo_sim>=0.3 non-muscle/pulley actuators unsupported by MuscleGeometry.
    m_tmp = spec.copy().compile()
    s_w = int(mujoco.mjtWrap.mjWRAP_SITE)
    ok_w = {
        s_w,
        int(mujoco.mjtWrap.mjWRAP_SPHERE),
        int(mujoco.mjtWrap.mjWRAP_CYLINDER),
    }
    drop_act_names: set[str] = set()
    for col in range(m_tmp.nu):
        t = int(m_tmp.actuator_trnid[col, 0])
        adr, num = int(m_tmp.tendon_adr[t]), int(m_tmp.tendon_num[t])
        typ = [int(x) for x in m_tmp.wrap_type[adr : adr + num]]
        valid_wraps = (
            len(typ) >= 2
            and typ[0] == s_w
            and typ[-1] == s_w
            and all(w in ok_w for w in typ)
            and all(
                not (typ[k] != s_w and typ[k + 1] != s_w) for k in range(len(typ) - 1)
            )
        )
        valid_act = (
            int(m_tmp.actuator_trntype[col]) == int(mujoco.mjtTrn.mjTRN_TENDON)
            and float(m_tmp.actuator_gear[col, 0]) == 1.0
            and not bool(m_tmp.actuator_forcelimited[col])
            and float(m_tmp.actuator_gainprm[col, 2]) > 0.0
            and float(m_tmp.actuator_biasprm[col, 2]) > 0.0
        )
        if not (valid_wraps and valid_act):
            drop_act_names.add(m_tmp.actuator(col).name)
    if drop_act_names:
        for act in list(spec.actuators):
            if act.name in drop_act_names:
                spec.delete(act)
    return spec


def _sample_state(gen: torch.Generator, batch: int, nq: int) -> tuple:
    from supermimic.rotations import quat_to_mat

    def rn(*s: int) -> torch.Tensor:
        return torch.randn(*s, generator=gen)

    return (
        rn(batch, 3),
        quat_to_mat(rn(batch, 4)),
        0.4 * rn(batch, 3),
        0.4 * rn(batch, 3),
        0.3 * rn(batch, nq),
        0.5 * rn(batch, nq),
    )


@pytest.mark.tier1
def test_accel_world_model_integrates_semi_implicit_targets_and_inference_copy() -> (
    None
):
    from supermimic.networks import WorldModel, inference_copy
    from supermimic.rotations import mm3, mv3, so3_exp

    gen = torch.Generator().manual_seed(7)
    nq, nm, nt = 6, 8, 4
    wm = WorldModel(nq, nm, nt, hidden=(32, 32), dt=0.02, substeps=5, kind="accel")
    assert wm.out_dim == 6 + nq
    assert wm.pos_coef == pytest.approx((5 + 1) / (2 * 5) * 0.02)

    S = _sample_state(gen, 5, nq)
    # Generate S1 via exact semi-implicit Euler accelerations so targets() + integrate() round-trip.
    av_l = 0.5 * torch.randn(5, 3, generator=gen)
    aw_l = 0.5 * torch.randn(5, 3, generator=gen)
    aq = 0.5 * torch.randn(5, nq, generator=gen)
    av, aw = mv3(S[1], av_l), mv3(S[1], aw_l)
    c = wm.pos_coef
    S1 = (
        S[0] + wm.dt * (S[2] + c * av),
        mm3(so3_exp(wm.dt * (S[3] + c * aw)), S[1]),
        S[2] + wm.dt * av,
        S[3] + wm.dt * aw,
        S[4] + wm.dt * (S[5] + c * aq),
        S[5] + wm.dt * aq,
    )
    targ = wm.targets(S, S1)
    S1_hat = wm.integrate(S, targ)
    for got, want in zip(S1_hat, S1, strict=True):
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)

    # Update normalizers with non-trivial statistics and verify fp32 + bf16 inference copies.
    act0 = torch.rand(5, nm, generator=gen)
    act1 = torch.rand(5, nm, generator=gen)
    wm.in_norm.update(wm.inputs(S, act0, act1))
    wm.out_norm.update(targ)
    wm.touch_norm.update(torch.rand(5, nt, generator=gen))
    nn_w = wm.net[-1].weight
    with torch.no_grad():
        nn_w.normal_(0.0, 0.05)

    inf_fp32 = inference_copy(wm, dtype=None)
    inf_bf16 = inference_copy(wm, dtype=torch.bfloat16)
    exc = torch.rand(5, 4, nm, generator=gen)
    states, acts, touches = inf_fp32.rollout(S, act0, exc)
    assert states[0].shape == (5, 4, 3)
    assert acts.shape == (5, 4, nm)
    assert touches.shape == (5, 4, nt)
    assert (touches >= 0).all()
    s_b, _ = inf_bf16.step(S, act0, act1)
    s_f, _ = inf_fp32.step(S, act0, act1)
    torch.testing.assert_close(s_b[0], s_f[0], atol=2e-2, rtol=2e-2)


@pytest.mark.tier1
@pytest.mark.parametrize("kind", ["increment", "accel"])
def test_rigidformer_world_model_se2_equivariance_and_muon(kind: str) -> None:
    from supermimic.activation import MuscleActivation
    from supermimic.kinematics import SkeletonKinematics
    from supermimic.muscle_geometry import MuscleGeometry
    from supermimic.networks import WorldModel, inference_copy
    from supermimic.optim import MuonAdam
    from supermimic.rotations import mm3, mv3

    from myosuite.envs.myo.tasks.mimic.tracking.models import model_cfg

    mj_body = _build_hinge_spec(False).compile()
    kin = SkeletonKinematics(model_cfg("myofullbody").tracked_bodies, "cpu", mj_body)
    MuscleGeometry(mj_body, "cpu")
    MuscleActivation(mj_body, 10, 0.002, "cpu")
    nq, nm = kin.num_q, int(mj_body.nu)
    wm = WorldModel(
        nq,
        nm,
        4,
        dt=0.02,
        substeps=5,
        kind=kind,
        backbone="rigidformer",
        kin=kin,
        mj_model=mj_body,
        rf_cfg={"dim": 48, "layers": 2, "heads": 4, "registers": 2, "muscle_dim": 16},
    )
    with torch.no_grad():
        wm.net.root_head.weight.normal_(0.0, 0.05)
        wm.net.joint_head.weight.normal_(0.0, 0.05)
        wm.net.touch_head.weight.normal_(0.0, 0.05)

    gen = torch.Generator().manual_seed(13)
    S = _sample_state(gen, 3, nq)
    act0 = torch.rand(3, nm, generator=gen)
    act1 = torch.rand(3, nm, generator=gen)

    # Apply horizontal translation (dx, dy, 0) and world yaw rotation Rz(psi).
    psi = torch.tensor([0.7, -1.2, 2.1])
    c, s, z, o = psi.cos(), psi.sin(), torch.zeros_like(psi), torch.ones_like(psi)
    Rz = torch.stack((c, -s, z, s, c, z, z, z, o), dim=-1).view(3, 3, 3)
    shift_xy = torch.tensor([[3.5, -2.0, 0.0], [-1.1, 4.2, 0.0], [0.8, 0.9, 0.0]])
    S_rot = (
        mv3(Rz, S[0]) + shift_xy,
        mm3(Rz, S[1]),
        mv3(Rz, S[2]),
        mv3(Rz, S[3]),
        S[4],
        S[5],
    )

    # Raw inputs (body features + 3D RoPE coords) are strictly SE(2)-invariant, and outputs are SE(2)-equivariant.
    torch.testing.assert_close(
        wm.inputs(S, act0, act1), wm.inputs(S_rot, act0, act1), atol=1e-5, rtol=1e-5
    )
    S1, t1 = wm.step(S, act0, act1)
    S1_rot, t1_rot = wm.step(S_rot, act0, act1)
    torch.testing.assert_close(
        S1_rot[0], mv3(Rz, S1[0]) + shift_xy, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(S1_rot[1], mm3(Rz, S1[1]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(S1_rot[2], mv3(Rz, S1[2]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(S1_rot[3], mv3(Rz, S1[3]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(S1_rot[4], S1[4], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(S1_rot[5], S1[5], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(t1_rot, t1, atol=1e-5, rtol=1e-5)

    # MuonAdam splits RigidFormer's 2D block weights onto Muon and the rest onto Adam.
    opt = MuonAdam(wm, None, lr=1e-3, fused=False, matrices=wm.net.hidden_matrices())
    loss = sum(t.square().mean() for t in S1) + t1.square().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

    inf = inference_copy(wm)
    s_inf, t_inf = inf.step(S, act0, act1)
    torch.testing.assert_close(
        s_inf[0], wm.step(S, act0, act1)[0][0], atol=1e-5, rtol=1e-5
    )
    assert t_inf.shape == (3, 4)


@pytest.mark.tier1
def test_loss_balancer_and_target_rate_penalty() -> None:
    from supermimic.supertrack import LossBalancer, target_rate_penalty

    bal = LossBalancer(3)
    assert not bal.calibrated
    terms = torch.tensor([2.0, 0.5, 4.0])
    counts = torch.tensor([10.0, 5.0, 20.0])
    bal.calibrate(terms, counts)
    assert bal.calibrated
    torch.testing.assert_close(bal.w * terms, torch.ones(3))
    torch.testing.assert_close(bal.scale, torch.tensor([0.2, 0.1, 0.2]))

    a_smooth = torch.tensor([[0.20, 0.20], [0.22, 0.21]])
    a_jerky = torch.tensor([[0.85, 0.10], [0.10, 0.90]])
    a_prev = torch.tensor([[0.20, 0.20], [0.20, 0.20]])
    assert bool(
        (
            target_rate_penalty(a_smooth, a_prev) < target_rate_penalty(a_jerky, a_prev)
        ).all()
    )


@pytest.mark.tier2
def test_supertrack_and_rigidformer_runner_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from myosuite.envs.myo.tasks.mimic.tracking import MIMIC_ENV_ID
    from myosuite.envs.myo.tasks.mimic.tracking.models import (
        MODELS,
        MimicModelCfg,
        compiled_body,
        compiled_scene,
        kinematic_fingerprint,
        model_cfg,
    )
    from myosuite.envs.myo.tasks.mimic.tracking.motion import write_synthetic_clip

    base = model_cfg("myofullbody")
    hinge_cfg = MimicModelCfg(
        name="myofullbody",
        build_spec=_build_hinge_spec,
        tracked_bodies=base.tracked_bodies,
        foot_bodies=base.foot_bodies,
        muscle_tendon_expr=base.muscle_tendon_expr,
    )
    monkeypatch.setitem(MODELS, "myofullbody", hinge_cfg)
    compiled_body.cache_clear()
    compiled_scene.cache_clear()
    kinematic_fingerprint.cache_clear()

    pytest.importorskip("mjlab")
    from mjlab.rl import RslRlVecEnvWrapper
    from supermimic.config import fast_runner_cfg, supertrack_runner_cfg
    from supermimic.env import TaskEnvCfg, task_overrides
    from supermimic.runner import SuperMimicRunner

    from myosuite import make_env
    from myosuite.core.config import EnvConfig

    clip_path = tmp_path / "clip0.npz"
    write_synthetic_clip(
        clip_path, compiled_body("myofullbody"), 6.0, 0.4, 0.05, 0, 0.0
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    task = TaskEnvCfg(motion_file=str(clip_path), num_envs=2, sampling_mode="start")
    overrides = {**task_overrides(task), "self_collisions": False}
    raw_env = make_env(
        EnvConfig(
            MIMIC_ENV_ID,
            backend="mjlab",
            num_envs=task.num_envs,
            ctrl_dt=task.ctrl_dt,
            task_kwargs={
                "model": "myofullbody",
                "preset": "supermimic",
                "overrides": overrides,
            },
        ),
        device=dev,
    )
    env = RslRlVecEnvWrapper(raw_env)
    try:
        log_dir = tmp_path / "run"
        cfg = supertrack_runner_cfg(
            wm_backbone="rigidformer",
            wm_mirror=0.0,
            rf_dim=32,
            rf_layers=1,
            rf_heads=4,
            rf_registers=2,
            rf_muscle_dim=16,
            enc_hidden=(32, 16, 16),
            dec_hidden=(32, 16, 16),
            critic_hidden=(16,),
            buffer_steps=24,
            warmup_steps=12,
            wm_batch=4,
            wm_batches=1,
            wm_only_iters=1,
            pi_batch=4,
            horizon=3,
            cuda_graph=False,
            collect_graph=False,
            overlap_collect=False,
            no_sync=False,
            fused_adam=False,
            compile=False,
            log_interval=1,
            save_interval=2,
            st_late_w=0.5,
        )
        runner = SuperMimicRunner(env, asdict(cfg), log_dir=str(log_dir), device=dev)
        runner.learn(num_learning_iterations=2)
        assert runner.bal_wm.calibrated and runner.bal_pi.calibrated
        ckpt = log_dir / "model_000002.pt"
        assert ckpt.is_file()

        # Reload into a default SuperMimic-configured runner and verify architecture/algo auto-switch.
        runner2 = SuperMimicRunner(
            env,
            asdict(
                fast_runner_cfg(
                    wm_mirror=0.0,
                    cuda_graph=False,
                    collect_graph=False,
                    overlap_collect=False,
                    no_sync=False,
                    fused_adam=False,
                    compile=False,
                )
            ),
            device=dev,
        )
        runner2.load(str(ckpt))
        assert runner2.cfg.algo == "supertrack"
        assert runner2.wm.kind == "accel"
        assert runner2.wm.backbone == "rigidformer"
        assert runner2.bal_wm.calibrated and runner2.bal_pi.calibrated
    finally:
        env.close()
        compiled_body.cache_clear()
        compiled_scene.cache_clear()
        kinematic_fingerprint.cache_clear()
