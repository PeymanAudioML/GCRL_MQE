# GCRL with MQE — Modern Flax NNX Implementation

**Author:** Peyman Poozesh ([@PeymanAudioML](https://github.com/PeymanAudioML))

This repository contains my updated implementation of offline **Goal-Conditioned Reinforcement Learning (GCRL)** algorithms on [OGBench](https://github.com/seohongpark/ogbench), with the primary contribution being **Multistep Quasimetric Estimation (MQE)**. The entire codebase has been migrated from legacy Flax Linen to **Flax NNX (≥ 0.8.4)** with **Orbax checkpointing**, making it compatible with the latest JAX/Flax ecosystem.

---

## What is MQE?

**Multistep Quasimetric Estimation (MQE)** is a goal-conditioned offline RL method that learns a quasimetric distance between states via two representation networks:

- **φ(s, a)** — state-action representation
- **ψ(s)** — state (goal) representation

The distance is computed using an **MRN (Metric Residual Network)** decomposition that combines a symmetric Euclidean component with an asymmetric L∞-based quasimetric, averaged across `K` independent components. The critic is trained with a **LINEX-style backup** that incorporates intermediate value goals, and the actor uses **DDPG+BC**.

Key MQE ideas:
- Multistep temporal backup via intermediate value goals sampled between the current state and the final goal
- Action invariance loss to keep φ(s, a) close to ψ(s)
- Off-diagonal backup weighting (`diag_backup`) to balance diagonal vs. off-diagonal gradient signal

---

## Supported Algorithms

| Agent | File | Description |
|-------|------|-------------|
| **MQE** | `agents/mqe.py` | Multistep Quasimetric Estimation (this work) |
| GCIQL | `agents/gciql.py` | Goal-conditioned IQL (AWR or DDPG+BC actor) |
| GCBC | `agents/gcbc.py` | Goal-conditioned Behavioral Cloning |
| GCIVL | `agents/gcivl.py` | Goal-conditioned IVL |
| HIQL | `agents/hiql.py` | Hierarchical IQL |
| CRL | `agents/crl.py` | Contrastive RL |
| QRL | `agents/qrl.py` | Quasimetric RL |
| CMD | `agents/cmd.py` | Contrastive MDP with distance |
| TMD | `agents/tmd.py` | Temporal Metric Distance |
| NGCSACBC | `agents/ngcsacbc.py` | Non-goal-conditioned SAC+BC |
| SAC | `agents/sac.py` | Soft Actor-Critic |

---

## Supported Environments (OGBench)

State-based:
- `pointmaze-{medium,large,giant,teleport}-{navigate,stitch}-v0`
- `antmaze-{medium,large,giant,teleport}-{navigate,stitch,explore}-v0`
- `humanoidmaze-{medium,large,giant}-{navigate,stitch}-v0`
- `antsoccer-{arena,medium}-{navigate,stitch}-v0`

Visual (pixel) observations:
- `visual-antmaze-{medium,large,giant,teleport}-{navigate,stitch}-v0`

---

## Flax NNX Migration

This codebase has been updated from the legacy Flax Linen (`flax.linen`) API to the new **Flax NNX** (`flax.nnx`) API. Key changes:

- All network modules (`MLP`, `GCActor`, `GCValue`, `GCBilinearValue`, etc.) are implemented as `nnx.Module` subclasses with explicit `rngs` arguments.
- A custom `TrainState` dataclass (registered as a JAX pytree) replaces `flax.training.train_state.TrainState`. It stores `graphdefs` (static) and `params` (dynamic) separately, enabling `jax.jit` and `jax.grad` to work correctly.
- `nnx.split` / `nnx.merge` are used internally by `TrainState` to split modules into their graph definitions and parameter states.
- Checkpointing uses **Orbax** (`orbax.checkpoint`) via `ocp.StandardCheckpointer`.
- The `ImpalaEncoder` uses a hand-written `_max_pool` function (via `jax.lax.reduce_window`) as a drop-in for `flax.linen.max_pool`.

Requirements: `flax >= 0.8.4`, `jax[cuda12] >= 0.4.26`, `orbax-checkpoint`.

---

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
ogbench
jax[cuda12] >= 0.4.26
flax >= 0.8.4
distrax >= 0.1.5
ml_collections
matplotlib
moviepy
wandb
```

For a local OGBench installation, replace `ogbench` with `pip install -e /path/to/ogbench`.

---

## Usage

### Basic training

```bash
python main.py \
  --env_name=antmaze-large-navigate-v0 \
  --agent=agents/mqe.py \
  --train_steps=1000000 \
  --eval_interval=50000 \
  --eval_episodes=50
```

### Run MQE on a specific environment

```bash
python main.py \
  --env_name=antmaze-large-navigate-v0 \
  --agent=agents/mqe.py \
  --agent.discount=0.995 \
  --agent.alpha=0.1 \
  --train_steps=1000000
```

### Override agent hyperparameters via flags

All agent config fields can be overridden with `--agent.<field>=<value>`:

```bash
python main.py \
  --env_name=pointmaze-medium-navigate-v0 \
  --agent=agents/gciql.py \
  --agent.alpha=0.003 \
  --agent.actor_loss=awr
```

### Visual environments (pixel observations)

```bash
python main.py \
  --env_name=visual-antmaze-medium-navigate-v0 \
  --agent=agents/mqe.py \
  --agent.encoder=impala_small \
  --agent.batch_size=256 \
  --train_steps=500000
```

Available encoders: `impala`, `impala_small`, `impala_large`, `impala_debug`.

### Restore a checkpoint

```bash
python main.py \
  --env_name=antmaze-large-navigate-v0 \
  --agent=agents/mqe.py \
  --restore_path="exp/OGBench/MyRun/sd000_*" \
  --restore_epoch=1000000
```

---

## Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--env_name` | `antmaze-large-navigate-v0` | OGBench environment name |
| `--agent` | `agents/gciql.py` | Agent config file |
| `--seed` | `0` | Random seed |
| `--train_steps` | `1000` | Number of training steps |
| `--eval_interval` | `100000` | Steps between evaluations |
| `--eval_episodes` | `20` | Episodes per task at evaluation |
| `--save_interval` | `1000000` | Steps between checkpoints |
| `--save_dir` | `exp/` | Output directory |
| `--run_group` | `Debug` | W&B run group |
| `--eval_on_cpu` | `1` | Run evaluation on CPU |

---

## MQE Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `discount` | `0.995` | Discount factor (also controls geometric goal sampling) |
| `lambda_` | `0.95` | Discount for intermediate value goal sampling |
| `next_state_sample` | `0.2` | Probability of using next state as intermediate goal |
| `alpha` | `0.1` | BC coefficient in DDPG+BC actor loss |
| `t` | `5.0` | LINEX clipping threshold |
| `diag_backup` | `0.5` | Weight of diagonal vs. off-diagonal backup |
| `components` | `8` | Number of MRN components |
| `latent_dim` | `512` | Representation dimension |
| `normalize_q_loss` | `True` | Normalize Q loss by absolute mean |

---

## Hyperparameter Reference

See `hyperparameters.sh` for the full list of recommended hyperparameters for every agent × environment combination used in OGBench experiments.

---

## Project Structure

```
.
├── main.py                  # Training loop
├── requirements.txt
├── hyperparameters.sh       # Full hyperparameter reference
├── agents/
│   ├── __init__.py          # Agent registry
│   ├── mqe.py               # MQE (main contribution)
│   ├── gciql.py             # GCIQL
│   ├── gcbc.py              # GCBC
│   ├── gcivl.py             # GCIVL
│   ├── hiql.py              # HIQL
│   ├── crl.py               # CRL
│   ├── qrl.py               # QRL
│   ├── cmd.py               # CMD
│   ├── tmd.py               # TMD
│   ├── ngcsacbc.py          # NGCSACBC
│   └── sac.py               # SAC
└── utils/
    ├── networks.py          # NNX network modules
    ├── flax_utils.py        # TrainState, save/restore
    ├── encoders.py          # IMPALA encoder, GCEncoder
    ├── datasets.py          # Dataset, GCDataset, HGCDataset
    ├── env_utils.py         # Environment wrappers
    ├── evaluation.py        # Evaluation loop
    └── log_utils.py         # CSV/W&B logging
```

---

## Logging

Training metrics are logged to:
- **Weights & Biases** (online by default — set `WANDB_MODE=disabled` to run offline)
- **CSV files** at `<save_dir>/train.csv` and `<save_dir>/eval.csv`
- **Flags** saved to `<save_dir>/flags.json`
