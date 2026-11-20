"""Benchmark DreamerV3 on GPU-accelerated MyoSuite MJX / Warp environments.

Mirrors `benchmarks/mjx_benchmark_PPO.py` so DreamerV3 world-model training can
be compared directly against Brax PPO on `MjxElbowPose*`, `MjxFingerPose*`, and
`MjxHandReach*` tasks using batched GPU `MJXDriver` rollouts.

Example:
  python benchmarks/mjx_benchmark_DreamerV3.py --env_name MjxElbowPoseRandom-v0 --impl warp --num_envs 64
"""

import argparse
import os
import pathlib
import timeit

import numpy as np
from dreamerv3 import main as dv3_main


def measure_num_env_training_steps(
    meta_seed=0,
    env_name="MjxElbowPoseRandom-v0",
    impl="jax",
    num_timesteps=500_000,
    num_envs=64,
    train_ratio=64.0,
    size_preset="size12m",
    platform="cuda",
    repeat=1,
    debug=False,
):
    """Measure wall-clock execution time for DreamerV3 + MJXDriver training."""
    ckpt_root = pathlib.Path("mjx_benchmark_DreamerV3_logs")
    ckpt_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=meta_seed)

    def run_benchmark():
        seed = int(rng.integers(0, 10000))
        run_dir = ckpt_root / f"{env_name}_{impl}_{num_envs}_seed{seed}_{{timestamp}}"
        configs = ["myosuite_mjx"]
        if size_preset != "size12m":
            configs.append(size_preset)
        if debug:
            configs.append("debug")

        argv = [
            "--configs",
            *configs,
            "--script",
            "train_mjx",
            "--task",
            f"myomjx_{env_name}",
            "--env.myomjx.impl",
            impl,
            "--logdir",
            str(run_dir),
            "--seed",
            str(seed),
            "--logger.outputs",
            "jsonl",
            "--run.steps",
            str(num_timesteps),
            "--run.envs",
            str(num_envs),
            "--run.train_ratio",
            str(train_ratio),
        ]
        if not debug:
            argv += ["--jax.platform", platform]
        dv3_main.main(argv)

    results = timeit.repeat(run_benchmark, number=1, repeat=repeat)
    print(
        f"Results for {num_envs} envs: DreamerV3 training for {num_timesteps} "
        f"total steps took {results} seconds (mean={np.mean(results):.2f}s)."
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DreamerV3 training for MyoSuite MJX environments."
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="MjxElbowPoseRandom-v0",
        help="MyoSuite MJX environment name.",
    )
    parser.add_argument(
        "--impl",
        type=str,
        default="warp",
        choices=("jax", "warp"),
        help="MJX physics backend implementation.",
    )
    parser.add_argument("--num_envs", type=int, default=64, help="Number of batched GPU envs.")
    parser.add_argument("--num_timesteps", type=int, default=500_000, help="Total env steps.")
    parser.add_argument("--train_ratio", type=float, default=64.0, help="Replay train ratio.")
    parser.add_argument(
        "--size",
        type=str,
        default="size12m",
        choices=("size1m", "size12m", "size25m", "size50m"),
    )
    parser.add_argument("--platform", type=str, default="cuda", choices=("cuda", "cpu", "tpu"))
    parser.add_argument("--repeat", type=int, default=1, help="Number of benchmark repeats.")
    parser.add_argument("--debug", action="store_true", help="Run fast CPU debug smoke test.")
    args = parser.parse_args()

    results = {}
    print(f"MyoSuite env {args.env_name} -- DreamerV3 MJX ({args.impl}, {args.num_envs} envs)")
    results[f"{args.env_name}_{args.impl}_{args.num_envs}"] = measure_num_env_training_steps(
        meta_seed=0,
        env_name=args.env_name,
        impl=args.impl,
        num_timesteps=args.num_timesteps,
        num_envs=args.num_envs,
        train_ratio=args.train_ratio,
        size_preset=args.size,
        platform=args.platform,
        repeat=args.repeat,
        debug=args.debug,
    )
    out_file = f"mjx_benchmark_DreamerV3_results_{args.env_name}_{args.impl}_{args.num_envs}.npy"
    np.save(out_file, results, allow_pickle=True)
    print(f"Saved results to {os.path.abspath(out_file)}")


if __name__ == "__main__":
    main()
