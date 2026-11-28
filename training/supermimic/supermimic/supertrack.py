# Copyright (c) MyoSuite Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0. Adapted from NDM Gym
# (github.com/Frank-ZY-Dou/ndm-gym, MIT; see LICENSE-ndm-gym.txt in training/supermimic/)
# and SuperTrack (Fussell, Bergamin, Holden, ACM TOG 2021).
"""SuperTrack world-model and critic-free policy losses for musculoskeletal tracking."""

from __future__ import annotations

import torch
from torch import nn

from supermimic.heads import StepCtx, TrackingHeads
from supermimic.kinematics import SkeletonKinematics
from supermimic.rotations import mat_to_6d, mm3, mv3, so3_log

WM_TERMS = ("pos", "vel", "rot", "ang")
LOCAL_TERMS = ("lpos", "lvel", "lrot", "lang", "hei", "up")
TRACK_TERMS = LOCAL_TERMS + ("rpos", "rrot")


class LossBalancer(nn.Module):
    """Self-calibrating term weights set on the first update so each term starts at 1.0."""

    def __init__(self, n_terms: int) -> None:
        super().__init__()
        self.register_buffer("w", torch.ones(n_terms))
        self.register_buffer("scale", torch.ones(n_terms))
        self.register_buffer("ready", torch.zeros((), dtype=torch.bool))
        self._calibrated = False

    @property
    def calibrated(self) -> bool:
        if not self._calibrated:
            self._calibrated = bool(self.ready)
        return self._calibrated

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self._calibrated = False

    @torch.no_grad()
    def calibrate(self, terms: torch.Tensor, n_elements: torch.Tensor) -> None:
        safe = terms.detach().float().clamp(min=1e-6)
        ok = torch.isfinite(safe).all()
        w = torch.where(ok, 1.0 / safe, self.w)
        sc = torch.where(
            ok, safe / n_elements.to(safe.dtype).clamp(min=1.0), self.scale
        )
        self.w.copy_(w)
        self.scale.copy_(sc)
        self.ready.fill_(True)
        self._calibrated = True


def local_body_features(
    bp: torch.Tensor,
    bR: torch.Tensor,
    bv: torch.Tensor,
    bw: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """SuperTrack root-local body features ``(lpos, lvel, lrot, lang, hei, up)``."""
    Rt = bR[..., :1, :, :].transpose(-1, -2)
    lp = mv3(Rt, bp - bp[..., :1, :])
    lv = mv3(Rt, bv)
    lr = mat_to_6d(mm3(Rt, bR))
    lw = mv3(Rt, bw)
    return lp, lv, lr, lw, bp[..., 2], bR[..., 0, 2, :]


def wm_window_l1_terms(
    kin: SkeletonKinematics,
    S_hat: tuple,
    S_true: tuple,
    *,
    mask: torch.Tensor,
    squared: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """World-space body ``(pos, vel, rot, ang)`` window errors ``(4,)``, element counts ``(4,)``, and last-step error."""
    bsz, win = S_true[0].shape[:2]
    flat_hat = tuple(x.reshape(bsz * win, *x.shape[2:]) for x in S_hat[:6])
    flat_true = tuple(x.reshape(bsz * win, *x.shape[2:]) for x in S_true[:6])
    ph, Rh, vh, wh = kin.forward(*flat_hat, all_bodies=True)
    with torch.no_grad():
        pt, Rt, vt, wt = kin.forward(*flat_true, all_bodies=True)
    norm = (lambda z: z.square()) if squared else (lambda z: z.abs())
    per_step = torch.stack(
        (
            norm(ph - pt).sum(dim=(-1, -2)),
            norm(vh - vt).sum(dim=(-1, -2)),
            norm(so3_log(mm3(Rt.transpose(-1, -2), Rh))).sum(dim=(-1, -2)),
            norm(wh - wt).sum(dim=(-1, -2)),
        ),
        dim=-1,
    ).view(bsz, win, 4)
    m = mask.unsqueeze(-1).float()
    denom = mask.sum().clamp(min=1.0)
    mean_terms = (per_step * m).sum(dim=(0, 1)) / denom
    nb = kin.num_bodies
    counts = torch.tensor(
        [3 * nb, 3 * nb, 3 * nb, 3 * nb], device=ph.device, dtype=torch.float32
    )
    return mean_terms, counts, per_step[:, -1, 0] / max(3 * nb, 1)


def supertrack_tracking_terms(
    heads: TrackingHeads,
    S: tuple,
    f: torch.Tensor,
    *,
    include_root: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-rollout SuperTrack L1 tracking terms ``(N, n_terms)`` and element counts ``(n_terms,)``."""
    ctx = StepCtx(S, f)
    bp, bR, bv, bw = heads.fk(ctx)
    with torch.no_grad():
        r_p = heads.root_p[f]
        r_R = heads.root_R[f]
        r_v = heads.motion.frames["root_qvel"][f, :3]
        r_w_body = heads.motion.frames["root_qvel"][f, 3:6]
        r_Rb = mm3(r_R, heads.kin.R_off.T)
        r_w = mv3(r_Rb, r_w_body)
        r_q = heads.q[f]
        cols = torch.as_tensor(heads.kin.info.indep_cols, device=f.device)
        r_qd = heads.ref_qd[f][:, cols]
        ref_fk = heads.kin.forward(r_p, r_R, r_v, r_w, r_q, r_qd)

    pred_local = local_body_features(bp, bR, bv, bw)
    ref_local = local_body_features(*ref_fk)
    errs = [
        (x - r).abs().flatten(1) for x, r in zip(pred_local, ref_local, strict=True)
    ]
    if include_root:
        errs.append((S[0] - r_p).abs())
        errs.append(so3_log(mm3(r_R.transpose(-1, -2), S[1])).abs())
    counts = torch.tensor(
        [e.shape[1] for e in errs], device=f.device, dtype=torch.float32
    )
    summed = torch.stack([e.sum(dim=-1) for e in errs], dim=-1)
    return summed, counts


def target_rate_penalty(
    a: torch.Tensor,
    a_prev: torch.Tensor,
    *,
    action_unit: float = 0.5,
) -> torch.Tensor:
    """Smooth-target penalty ``((a - a_prev) / action_unit)^2`` averaged over actuators ``(N,)``."""
    return ((a - a_prev) / action_unit).square().mean(dim=-1)
