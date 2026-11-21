# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Like-for-like multi-seed benchmark comparing DreamerV3 against PPO on MyoSuite."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import platform as py_platform
import time
from typing import Any

import numpy as np
import psutil

from myosuite_dreamerv3.compat import import_myosuite
from myosuite_dreamerv3.train import train_cpu


def evaluate_sb3_ppo(
    model: Any,
    env_id: str,
    episodes: int = 10,
    seed: int = 0,
    features: Sequence[Any] = (),
) -> dict[str, float]:
    """Evaluate a Stable-Baselines3 PPO model deterministically on ``env_id``."""
    myosuite = import_myosuite()
    returns: list[float] = []
    solved_flags: list[float] = []
    lengths: list[int] = []

    for ep in range(episodes):
        if features:
            from myosuite.core.config import EnvConfig
            from myosuite.utils.feature_cli import feature_spec

            parsed = tuple(
                feature_spec(f) if isinstance(f, str) else f for f in features
            )
            env = myosuite.make_env(EnvConfig(env_id=env_id, features=parsed))
        else:
            env = myosuite.make_env(env_id)
        try:
            obs, _ = env.reset(seed=int(seed + ep))
            done = False
            ep_ret = 0.0
            ep_solved = False
            ep_len = 0
            while not done:
                act, _ = model.predict(obs, deterministic=True)
                obs, rew, term, trunc, info = env.step(act)
                ep_ret += float(rew)
                rwd_dict = info.get("rwd_dict", {})
                solved = info.get("solved", rwd_dict.get("solved", 0.0))
                ep_solved = ep_solved or (float(np.asarray(solved).mean()) > 0.5)
                ep_len += 1
                done = bool(term or trunc)
            returns.append(ep_ret)
            solved_flags.append(1.0 if ep_solved else 0.0)
            lengths.append(ep_len)
        finally:
            env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "solved_rate": float(np.mean(solved_flags)),
        "mean_length": float(np.mean(lengths)),
    }


def run_ppo_arm(
    env_id: str,
    steps: int,
    seed: int,
    eval_episodes: int = 10,
    features: Sequence[Any] = (),
) -> dict[str, Any]:
    """Train and evaluate Stable-Baselines3 PPO on ``env_id`` for ``steps``."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    myosuite = import_myosuite()

    def _make_one() -> Any:
        if features:
            from myosuite.core.config import EnvConfig
            from myosuite.utils.feature_cli import feature_spec

            parsed = tuple(
                feature_spec(f) if isinstance(f, str) else f for f in features
            )
            env = myosuite.make_env(EnvConfig(env_id=env_id, features=parsed))
        else:
            env = myosuite.make_env(env_id)
        env.reset(seed=int(seed))
        return env

    vec_env = DummyVecEnv([_make_one])
    n_steps = min(512, max(64, steps // 2))
    batch_size = min(64, n_steps)
    proc = psutil.Process()
    rss_before = proc.memory_info().rss

    try:
        t0 = time.perf_counter()
        model = PPO(
            "MlpPolicy",
            vec_env,
            seed=int(seed),
            n_steps=n_steps,
            batch_size=batch_size,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=int(steps))
        wall_time = time.perf_counter() - t0
        rss_peak = max(rss_before, proc.memory_info().rss)
        eval_stats = evaluate_sb3_ppo(
            model,
            env_id=env_id,
            episodes=eval_episodes,
            seed=seed + 10_000,
            features=features,
        )
    finally:
        vec_env.close()

    return {
        "method": "PPO",
        "env_id": env_id,
        "seed": seed,
        "steps": steps,
        "wall_time_s": wall_time,
        "peak_rss_mb": rss_peak / (1024.0 * 1024.0),
        "eval": eval_stats,
    }


def run_dreamerv3_arm(
    env_id: str,
    steps: int,
    seed: int,
    logroot: Path,
    size: str = "size1m",
    platform: str = "cpu",
    envs: int = 4,
    train_ratio: float = 256.0,
    eval_episodes: int = 10,
    features: Sequence[Any] = (),
    debug: bool = False,
) -> dict[str, Any]:
    """Train and evaluate DreamerV3 on ``env_id`` for ``steps``."""
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    run_dir = logroot / f"dreamerv3_{env_id}_seed{seed}"
    res = train_cpu(
        env_id=env_id,
        steps=steps,
        envs=envs,
        train_ratio=train_ratio,
        size=size,
        platform=platform,
        seed=seed,
        logdir=str(run_dir),
        outputs=("jsonl",),
        script="train",
        features=features,
        debug=debug,
        eval_episodes=eval_episodes,
    )
    rss_peak = max(rss_before, proc.memory_info().rss)
    return {
        "method": f"DreamerV3 ({'debug' if debug else size})",
        "env_id": env_id,
        "seed": seed,
        "steps": steps,
        "wall_time_s": res["wall_time_s"],
        "peak_rss_mb": rss_peak / (1024.0 * 1024.0),
        "eval": res["eval"],
    }


def summarize_arms(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-seed runs into ``mean ± std`` rows grouped by ``(env_id, method)``."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault((rec["env_id"], rec["method"]), []).append(rec)

    summary: list[dict[str, Any]] = []
    for (env_id, method), items in grouped.items():
        rets = [r["eval"]["mean_return"] for r in items]
        solves = [r["eval"]["solved_rate"] for r in items]
        times = [r["wall_time_s"] for r in items]
        mems = [r["peak_rss_mb"] for r in items]
        summary.append(
            {
                "env_id": env_id,
                "method": method,
                "seeds": [r["seed"] for r in items],
                "steps": items[0]["steps"],
                "return_mean": float(np.mean(rets)),
                "return_std": float(np.std(rets)),
                "solved_mean": float(np.mean(solves)),
                "solved_std": float(np.std(solves)),
                "wall_time_mean_s": float(np.mean(times)),
                "wall_time_std_s": float(np.std(times)),
                "peak_rss_mean_mb": float(np.mean(mems)),
            }
        )
    return summary


def run_benchmark(
    env_ids: Sequence[str] = ("myoElbowPose1D6MRandom-v0", "myoHandPoseRandom-v0"),
    seeds: Sequence[int] = (0, 1, 2),
    steps: int = 8_000,
    eval_episodes: int = 10,
    size: str = "size1m",
    platform: str = "cpu",
    envs: int = 4,
    train_ratio: float = 256.0,
    out_path: str | Path = "dreamerv3_vs_ppo_benchmark.json",
    logroot: str | Path = "dreamerv3_bench_logs",
    debug: bool = False,
) -> dict[str, Any]:
    """Run multi-seed PPO vs. DreamerV3 benchmark across ``env_ids``."""
    logroot_path = Path(logroot)
    logroot_path.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for env_id in env_ids:
        for seed in seeds:
            ppo_rec = run_ppo_arm(
                env_id=env_id,
                steps=steps,
                seed=seed,
                eval_episodes=eval_episodes,
            )
            records.append(ppo_rec)
            dv3_rec = run_dreamerv3_arm(
                env_id=env_id,
                steps=steps,
                seed=seed,
                logroot=logroot_path,
                size=size,
                platform=platform,
                envs=envs,
                train_ratio=train_ratio,
                eval_episodes=eval_episodes,
                debug=debug,
            )
            records.append(dv3_rec)

    payload: dict[str, Any] = {
        "hardware": {
            "platform": py_platform.platform(),
            "processor": py_platform.processor(),
            "python": py_platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
        },
        "summary": summarize_arms(records),
        "runs": records,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """CLI entrypoint for ``dreamerv3-bench``."""
    parser = argparse.ArgumentParser(
        description="Run multi-seed PPO vs. DreamerV3 benchmark on MyoSuite tasks."
    )
    parser.add_argument(
        "--envs-list",
        nargs="+",
        default=["myoElbowPose1D6MRandom-v0", "myoHandPoseRandom-v0"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=8_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--size", type=str, default="size1m")
    parser.add_argument(
        "--platform", type=str, default="cpu", choices=("cpu", "cuda", "tpu")
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=256.0)
    parser.add_argument("--out", type=str, default="dreamerv3_vs_ppo_benchmark.json")
    parser.add_argument("--logroot", type=str, default="dreamerv3_bench_logs")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    result = run_benchmark(
        env_ids=args.envs_list,
        seeds=args.seeds,
        steps=args.steps,
        eval_episodes=args.eval_episodes,
        size=args.size,
        platform=args.platform,
        envs=args.num_envs,
        train_ratio=args.train_ratio,
        out_path=args.out,
        logroot=args.logroot,
        debug=args.debug,
    )
    for row in result["summary"]:
        print(
            f"{row['env_id']} | {row['method']}: "
            f"return={row['return_mean']:.2f}±{row['return_std']:.2f}, "
            f"solved={row['solved_mean']*100:.1f}±{row['solved_std']*100:.1f}%, "
            f"wall={row['wall_time_mean_s']:.1f}s"
        )
    return result


if __name__ == "__main__":
    main()
