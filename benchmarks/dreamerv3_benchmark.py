"""Train and evaluate DreamerV3 on MyoSuite MuJoCo / Gymnasium tasks.

Supports all MyoBase, MyoChallenge, MyoDM, and muscle-condition variants
(sarcopenia, fatigue, reafferentation), in proprioceptive or visual mode.

Example:
  python benchmarks/dreamerv3_benchmark.py --env_name myoElbowPose1D6MRandom-v0 --steps 500000
"""

import argparse
import importlib.util
import pathlib
import time

from dreamerv3 import main as dv3_main


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate DreamerV3 on MyoSuite Gymnasium environments."
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="myoElbowPose1D6MRandom-v0",
        help="MyoSuite Gymnasium environment ID (e.g. myoHandBaodingBallsP1-v1).",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="myosuite_proprio",
        choices=("myosuite_proprio", "myosuite_vision", "size1m", "size12m", "size25m", "size50m"),
        help="DreamerV3 model/environment preset from configs.yaml.",
    )
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total environment steps.")
    parser.add_argument("--envs", type=int, default=16, help="Parallel training environments.")
    parser.add_argument("--eval_envs", type=int, default=4, help="Parallel evaluation environments.")
    parser.add_argument("--train_ratio", type=float, default=512.0, help="Replay train ratio.")
    parser.add_argument("--vision", action="store_true", help="Include rendered RGB observations.")
    parser.add_argument("--platform", type=str, default="cuda", choices=("cuda", "cpu", "tpu"))
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--logdir",
        type=str,
        default="",
        help="Output log directory (defaults to ./dreamerv3_logs/<env_name>_<timestamp>).",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="train_eval",
        choices=("train", "train_eval", "eval_only"),
    )
    parser.add_argument(
        "--outputs",
        type=str,
        default="jsonl,scope" if importlib.util.find_spec("scope") else "jsonl",
        help="Comma-separated DreamerV3 logger outputs (e.g. jsonl,scope,wandb,tensorboard).",
    )
    parser.add_argument("--debug", action="store_true", help="Use tiny debug network & CPU settings.")
    return parser.parse_args()


def run(args):
    configs = [args.preset]
    if args.preset not in ("myosuite_proprio", "myosuite_vision"):
        configs = ["myosuite_proprio", args.preset]
    if args.debug:
        configs.append("debug")

    logdir = args.logdir or str(
        pathlib.Path("dreamerv3_logs") / f"{args.env_name}_{{timestamp}}"
    )
    outputs = [x.strip() for x in args.outputs.split(",") if x.strip()]
    argv = [
        "--configs",
        *configs,
        "--task",
        f"myosuite_{args.env_name}",
        "--script",
        args.script,
        "--logdir",
        logdir,
        "--logger.outputs",
        *outputs,
        "--seed",
        str(args.seed),
        "--run.steps",
        str(args.steps),
        "--run.envs",
        str(args.envs),
        "--run.eval_envs",
        str(args.eval_envs),
        "--run.train_ratio",
        str(args.train_ratio),
        "--env.myosuite.image",
        str(bool(args.vision or args.preset == "myosuite_vision")),
    ]
    if not args.debug:
        argv += ["--jax.platform", args.platform]

    start = time.perf_counter()
    dv3_main.main(argv)
    elapsed = time.perf_counter() - start
    print(f"Completed {args.steps} steps on {args.env_name} in {elapsed:.2f}s.")
    return elapsed


if __name__ == "__main__":
    run(parse_args())
