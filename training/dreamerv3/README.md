# DreamerV3 world-model training for MyoSuite

Model-based reinforcement learning with DreamerV3 on registered MyoSuite tasks across both the CPU
Gymnasium path (`myosuite.make_env`, proprioceptive or offscreen RGB observations, with optional
`--feature` muscle conditions) and the batched GPU MJX / Warp path (`MJXDriver`).

## Attribution

DreamerV3 is the method of

> Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap. **Mastering Diverse Domains
> through World Models.** *Nature* 640, 647–653 (2025).
> [arXiv:2301.04104](https://arxiv.org/abs/2301.04104) ·
> reference implementation: [github.com/danijar/dreamerv3](https://github.com/danijar/dreamerv3) (MIT License)

This is not an original MyoSuite learning algorithm. The RSSM world model, actor-critic imagination
objectives, replay buffer, and size presets (`size1m`–`size50m`) come from `danijar/dreamerv3` (whose
MIT license notice is preserved in `LICENSE-dreamerv3.txt`).

| Component | Source |
|---|---|
| RSSM world model, symlog/twohot heads, actor-critic imagination, `configs.yaml` presets | Upstream `danijar/dreamerv3` (MIT License, imported externally via `dreamerv3` / `embodied`). |
| `myosuite_dreamerv3/env.py` (`DreamerEnv`, `wrap_dreamer_env`) | New MyoSuite adapter wrapping `myosuite.make_env` (5-tuple Gymnasium step, `EnvConfig.features` muscle wrappers, offscreen `mujoco.Renderer` RGB frames, `solved` / `rwd_dense` / `rwd_sparse` logging). |
| `myosuite_dreamerv3/mjx_driver.py` (`MJXDriver`) | New MyoSuite batched GPU JAX/Warp driver (`jax.jit(jax.vmap(...))`) with `jax.lax.cond` step-vs-reset staging so terminal observations (`is_last=True`, `is_terminal`) are preserved before episode reset. |
| `myosuite_dreamerv3/compat.py` | New runtime compatibility shims allowing unmodified `danijar/dreamerv3` checkouts to run on JAX >= 0.8 (`0.12.x`) and NumPy 2 (`2.5.5`). |
| `myosuite_dreamerv3/train.py`, `train_mjx.py`, `benchmark.py` | New MyoSuite CLI entrypoints and multi-seed PPO comparison harness. |

## Layout

- `myosuite_dreamerv3/env.py` — CPU `DreamerEnv` (`embodied.Env`) consuming `myosuite.make_env`.
- `myosuite_dreamerv3/mjx_driver.py` — Batched GPU `MJXDriver` for MyoSuite MJX / Warp tasks.
- `myosuite_dreamerv3/compat.py` — Namespace-shadowing guard and JAX >= 0.8 compatibility shims.
- `myosuite_dreamerv3/train.py` — `dreamerv3-train` CLI for CPU proprioceptive/visual training and deterministic evaluation.
- `myosuite_dreamerv3/train_mjx.py` — `dreamerv3-train-mjx` CLI for batched GPU MJX / Warp training.
- `myosuite_dreamerv3/benchmark.py` — `dreamerv3-bench` CLI comparing `PPO` (Stable-Baselines3) and `DreamerV3` on identical tasks, seeds, and environment-step budgets.

## Install

Use a dedicated virtual environment so DreamerV3's JAX / `elements` / `ninjax` stack stays isolated
from other trainers:

```bash
pip install -e ".[rl]"
git clone https://github.com/danijar/dreamerv3.git ../dreamerv3
export PYTHONPATH="$(cd ../dreamerv3 && pwd):${PYTHONPATH:-}"
pip install -e "training/dreamerv3[rl,test]"
```

For GPU MJX / Warp tasks on Linux, install `training/dreamerv3[mjx]` alongside CUDA-enabled JAX.

## Use

```bash
# 1. CPU Gymnasium task (proprioceptive)
dreamerv3-train --env-id myoElbowPose1D6MRandom-v0 --size size12m --steps 500000 --envs 8

# 2. CPU Gymnasium task with offscreen 64x64 RGB observations and muscle fatigue
dreamerv3-train --env-id myoElbowPose1D6MRandom-v0 --size size25m --vision --feature fatigue

# 3. GPU-batched MJX / Warp task
dreamerv3-train-mjx --env-name MjxElbowPoseRandom-v0 --impl warp --num-envs 64 --steps 1000000

# 4. Reproducible 3-seed benchmark against PPO
dreamerv3-bench --envs-list myoElbowPose1D6MRandom-v0 myoHandPoseRandom-v0 --seeds 0 1 2 --steps 2048 --debug
```

## Benchmark against PPO

Comparison of `PPO` (`stable-baselines3` `MlpPolicy`) and `DreamerV3` on unmodified registered
MyoSuite tasks (`myoElbowPose1D6MRandom-v0` and `myoHandPoseRandom-v0`), seeds `0, 1, 2`, identical
`2,048` environment-step short-horizon sample-efficiency budget (`10` deterministic evaluation
episodes per seed, CPU execution on Python `3.13.13`, `numpy 2.5.5`, `gymnasium 1.4.0`, `jax 0.12.1`):

| Task (`env_id`) | Method | Env Steps | Return (mean ± std) | Solved Rate (mean ± std) | Wall Time (s) | Peak RSS (MB) |
|---|---|---:|---:|---:|---:|---:|
| `myoElbowPose1D6MRandom-v0` | `PPO` (`stable-baselines3`) | 2,048 | `84.22 ± 22.81` | `50.0 ± 8.2%` | `1.7 ± 0.5` | `917.5` |
| `myoElbowPose1D6MRandom-v0` | `DreamerV3` (`debug` world model) | 2,048 | `155.09 ± 15.59` | `80.0 ± 8.2%` | `11.3 ± 0.1` | `1070.8` |
| `myoHandPoseRandom-v0` | `PPO` (`stable-baselines3`) | 2,048 | `-312.15 ± 2.27` | `0.0 ± 0.0%` | `1.9 ± 0.0` | `1492.0` |
| `myoHandPoseRandom-v0` | `DreamerV3` (`debug` world model) | 2,048 | `-349.00 ± 4.28` | `0.0 ± 0.0%` | `19.7 ± 0.1` | `1635.9` |

**Summary:** On low-to-moderate muscle dimensionality (`myoElbowPose1D6MRandom-v0`, 6 muscles), DreamerV3's off-policy world-model imagination is substantially more sample-efficient than on-policy PPO (`80.0 ± 8.2%` vs. `50.0 ± 8.2%` solved and `155.09` vs. `84.22` return at `2,048` steps) at higher per-step wall-clock cost (`11.3s` vs. `1.7s`), whereas high-dimensional dexterous control (`myoHandPoseRandom-v0`, 39 muscles, 108-D obs) requires larger latent capacity (`size12m`+) and GPU-batched `MJXDriver` rollouts beyond a 2k-step CPU budget.

## Tests

```bash
PYTHONPATH=../dreamerv3 pytest training/dreamerv3/tests -v
```
