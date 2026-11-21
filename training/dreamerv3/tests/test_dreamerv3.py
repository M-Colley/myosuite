# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit and smoke tests for myosuite-dreamerv3."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from myosuite_dreamerv3.benchmark import run_benchmark
from myosuite_dreamerv3.env import DreamerEnv, wrap_dreamer_env
from myosuite_dreamerv3.mjx_driver import MJXDriver
from myosuite_dreamerv3.train import train_cpu
from myosuite_dreamerv3.train_mjx import (
    build_dreamer_config,
    make_agent_and_factories,
    run_mjx_training_loop,
)


class _ToyMjxState(NamedTuple):
    obs: dict[str, jax.Array]
    reward: jax.Array
    done: jax.Array
    metrics: dict[str, jax.Array]


class _ToyMjxEnv:
    """Minimal JAX-traceable MJX-compatible environment for fast contract verification."""

    action_size = 4

    class _Cfg:
        max_episode_steps = 3

    _config = _Cfg()

    def reset(self, rng: jax.Array) -> _ToyMjxState:
        obs = jax.random.uniform(rng, (6,), minval=-0.1, maxval=0.1)
        return _ToyMjxState(
            obs={"state": obs},
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
            metrics={"pose_err": jnp.float32(0.1)},
        )

    def step(self, state: _ToyMjxState, action: jax.Array) -> _ToyMjxState:
        next_obs = state.obs["state"] + 0.5 * jnp.pad(action, (0, 2))
        err = jnp.mean(jnp.square(next_obs))
        done = jnp.where(err > 0.8, jnp.float32(1.0), jnp.float32(0.0))
        return _ToyMjxState(
            obs={"state": next_obs},
            reward=-err,
            done=done,
            metrics={"pose_err": err},
        )


@pytest.mark.tier1
def test_dreamer_env_spaces_and_seeded_reproducibility() -> None:
    env_a = wrap_dreamer_env(DreamerEnv("myoElbowPose1D6MRandom-v0", seed=42))
    env_b = wrap_dreamer_env(DreamerEnv("myoElbowPose1D6MRandom-v0", seed=42))
    try:
        assert "proprio" in env_a.obs_space
        assert env_a.act_space["action"].low.tolist() == [-1.0] * 6
        assert env_a.act_space["action"].high.tolist() == [1.0] * 6

        zero_act = np.zeros(env_a.act_space["action"].shape, np.float32)
        obs_a0 = env_a.step({"action": zero_act, "reset": np.bool_(True)})
        obs_b0 = env_b.step({"action": zero_act, "reset": np.bool_(True)})
        assert bool(obs_a0["is_first"]) is True
        assert bool(obs_a0["is_last"]) is False
        np.testing.assert_allclose(obs_a0["proprio"], obs_b0["proprio"], atol=0.0)

        act = np.linspace(-0.5, 0.5, 6, dtype=np.float32)
        obs_a1 = env_a.step({"action": act, "reset": np.bool_(False)})
        obs_b1 = env_b.step({"action": act, "reset": np.bool_(False)})
        np.testing.assert_allclose(obs_a1["proprio"], obs_b1["proprio"], atol=0.0)
        assert obs_a1["reward"] == pytest.approx(obs_b1["reward"])
    finally:
        env_a.close()
        env_b.close()


@pytest.mark.tier1
def test_dreamer_env_features_and_offscreen_render() -> None:
    env = wrap_dreamer_env(
        DreamerEnv(
            "myoElbowPose1D6MRandom-v0",
            proprio=True,
            image=True,
            size=(64, 64),
            seed=7,
            features=("fatigue",),
        )
    )
    try:
        assert env.obs_space["image"].shape == (64, 64, 3)
        obs = env.step(
            {
                "action": np.zeros(env.act_space["action"].shape, np.float32),
                "reset": np.bool_(True),
            }
        )
        assert obs["image"].shape == (64, 64, 3)
        assert obs["image"].dtype == np.uint8
        assert int(obs["image"].max()) > 0
    finally:
        env.close()


@pytest.mark.tier1
def test_mjx_driver_preserves_terminal_observation_before_reset() -> None:
    driver = MJXDriver(_ToyMjxEnv(), num_envs=2, max_episode_steps=2, seed=0)
    transitions: list[tuple[int, dict[str, np.ndarray]]] = []
    driver.on_step(lambda trn, wid: transitions.append((wid, trn)))

    def _const_policy(carry, obs):
        n = len(obs["is_first"])
        return carry, {"action": np.ones((n, 4), dtype=np.float32)}, {}

    driver.reset()
    driver(_const_policy, steps=8)

    worker0 = [trn for wid, trn in transitions if wid == 0]
    assert [bool(t["is_first"]) for t in worker0] == [True, False, False, True]
    assert [bool(t["is_last"]) for t in worker0] == [False, False, True, False]
    assert np.linalg.norm(worker0[2]["state"]) > 0.5
    assert float(worker0[3]["reward"]) == 0.0


@pytest.mark.tier2
def test_train_cpu_and_mjx_smoke(tmp_path: Path) -> None:
    cpu_res = train_cpu(
        env_id="myoElbowPose1D6MRandom-v0",
        steps=40,
        envs=2,
        train_ratio=32.0,
        size="size1m",
        platform="cpu",
        seed=0,
        logdir=str(tmp_path / "cpu_run"),
        debug=True,
        eval_episodes=2,
    )
    assert "eval" in cpu_res
    assert np.isfinite(cpu_res["eval"]["mean_return"])

    mjx_cfg = build_dreamer_config(
        size="size1m",
        platform="cpu",
        seed=0,
        steps=40,
        envs=2,
        train_ratio=32.0,
        logdir=tmp_path / "mjx_run",
        debug=True,
    )
    (tmp_path / "mjx_run").mkdir(parents=True, exist_ok=True)
    driver = MJXDriver(_ToyMjxEnv(), num_envs=2, max_episode_steps=5, seed=0)
    agent, make_replay, make_stream, make_logger = make_agent_and_factories(
        mjx_cfg, driver.obs_space, driver.act_space
    )
    mjx_res = run_mjx_training_loop(
        driver, agent, make_replay, make_stream, make_logger, mjx_cfg
    )
    assert mjx_res["steps"] >= 40


@pytest.mark.tier2
def test_ppo_vs_dreamerv3_benchmark_smoke(tmp_path: Path) -> None:
    out_json = tmp_path / "bench.json"
    payload = run_benchmark(
        env_ids=("myoElbowPose1D6MRandom-v0",),
        seeds=(0,),
        steps=128,
        eval_episodes=2,
        size="size1m",
        platform="cpu",
        envs=2,
        train_ratio=16.0,
        out_path=out_json,
        logroot=tmp_path / "logs",
        debug=True,
    )
    assert out_json.is_file()
    assert len(payload["summary"]) == 2
