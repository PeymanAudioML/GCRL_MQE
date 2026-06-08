"""Lightweight validation for the native flax.nnx MQE refactor.

Runs on CPU with a small synthetic GCDataset (no env / MuJoCo needed) and
checks the full MQE path:

    1. import works
    2. GCDataset sampling works (produces the MQE batch fields)
    3. agent creation works
    4. one forward total_loss works
    5. one update step works (and actually mutates parameters)
    6. all required info keys are present
    7. no NaNs/Infs in loss or metrics
    8. sample_actions / get_distance / to_device work

Usage:  python debug_mqe.py
"""

import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# 1. Import works.
from agents.mqe import MQEAgent, get_config
from utils.datasets import Dataset, GCDataset


def build_synthetic_gcdataset(config, n_traj=20, traj_len=50, obs_dim=29, act_dim=8):
    """Build a tiny compact GCDataset with valid trajectory boundaries."""
    n = n_traj * traj_len
    observations = np.random.randn(n, obs_dim).astype(np.float32)
    actions = np.random.uniform(-1.0, 1.0, size=(n, act_dim)).astype(np.float32)

    terminals = np.zeros(n, dtype=np.float32)
    terminals[traj_len - 1 :: traj_len] = 1.0  # last step of each trajectory

    # Last state of each trajectory has no valid next state (compact dataset).
    valids = np.ones(n, dtype=np.float32)
    valids[traj_len - 1 :: traj_len] = 0.0

    dataset = Dataset.create(
        observations=observations,
        actions=actions,
        terminals=terminals,
        valids=valids,
    )
    return GCDataset(dataset, config)


def check_finite(name, info):
    bad = []
    for k, v in info.items():
        v = np.asarray(v)
        if not np.all(np.isfinite(v)):
            bad.append(k)
    assert not bad, f'[{name}] non-finite values in: {bad}'


def main():
    rng = jax.random.PRNGKey(0)
    np.random.seed(0)

    config = get_config()

    # 2. Dataset sampling works.
    train_dataset = build_synthetic_gcdataset(config)
    example_batch = train_dataset.sample(1)
    required_fields = [
        'observations', 'actions', 'next_observations',
        'value_goals', 'intermediate_value_goals',
        'intermediate_value_goals_offsets', 'value_goals_offsets',
        'actor_goals', 'masks', 'rewards',
    ]
    missing = [f for f in required_fields if f not in example_batch]
    assert not missing, f'batch missing fields: {missing}'
    print('[2] GCDataset.sample(1) OK; batch fields present:', sorted(example_batch.keys()))

    # 3. Agent creation works.
    agent = MQEAgent.create(
        0, example_batch['observations'], example_batch['actions'], config
    )
    assert hasattr(agent, 'model') and hasattr(agent, 'optimizer')
    assert agent.model.actor is not None and agent.model.phi is not None and agent.model.psi is not None
    print('[3] MQEAgent.create() OK (model: actor/phi/psi, nnx.Optimizer present)')

    # critic_loss assumes the sampled batch == config.batch_size (it draws the
    # next-state Bernoulli with shape (config.batch_size,)). Sample accordingly.
    batch = train_dataset.sample(config['batch_size'])

    # 4. Forward loss works.
    loss, info = agent.total_loss(agent.model, batch, rng)
    assert np.isfinite(float(loss)), f'total_loss not finite: {loss}'
    check_finite('total_loss', info)
    print(f'[4] total_loss OK; loss={float(loss):.4f}, #metrics={len(info)}')

    # 6. Required info keys exist.
    required_keys = [
        'critic/critic_loss', 'critic/backup_loss', 'critic/action_invariance_loss',
        'actor/actor_loss', 'actor/q_loss', 'actor/bc_loss',
    ]
    missing_keys = [k for k in required_keys if k not in info]
    assert not missing_keys, f'missing info keys: {missing_keys}'
    print('[6] All required info keys present.')

    # 5. One update step works and mutates parameters.
    before = float(jax.tree_util.tree_leaves(nnx.state(agent.model, nnx.Param))[0].sum())
    step_before = int(agent.optimizer.step.value)
    agent, update_info = agent.update(batch)
    step_after = int(agent.optimizer.step.value)
    after = float(jax.tree_util.tree_leaves(nnx.state(agent.model, nnx.Param))[0].sum())
    assert step_after == step_before + 1, f'optimizer step not advanced: {step_before}->{step_after}'
    assert before != after, 'parameters did not change after update'
    check_finite('update', update_info)
    for k in ['grad/max', 'grad/min', 'grad/norm']:
        assert k in update_info, f'missing grad metric: {k}'
    print(f'[5] update() OK; step {step_before}->{step_after}, params mutated, all metrics finite.')

    # 8. sample_actions / get_distance / to_device.
    obs = batch['observations']
    goals = batch['actor_goals']
    actions = agent.sample_actions(obs, goals, seed=jax.random.PRNGKey(1), temperature=0.0)
    assert actions.shape == batch['actions'].shape, actions.shape
    assert np.all(np.isfinite(np.asarray(actions)))
    print(f'[8a] sample_actions OK; shape={actions.shape}')

    d = agent.get_distance(obs, goals, batch['actions'])
    assert np.all(np.isfinite(np.asarray(d)))
    print(f'[8b] get_distance OK; shape={np.asarray(d).shape}')

    cpu = jax.devices('cpu')[0]
    eval_agent = agent.to_device(cpu)
    a2 = eval_agent.sample_actions(obs, goals, seed=jax.random.PRNGKey(2), temperature=0.0)
    assert a2.shape == batch['actions'].shape
    print('[8c] to_device + eval sample_actions OK')

    # Run a few more updates to confirm stability (no NaNs creeping in).
    for _ in range(5):
        agent, update_info = agent.update(train_dataset.sample(config['batch_size']))
    check_finite('update(after 5 steps)', update_info)
    print(f"[7] 6 updates total, still finite; "
          f"critic_loss={float(update_info['critic/critic_loss']):.4f}, "
          f"actor_loss={float(update_info['actor/actor_loss']):.4f}")

    print('\nALL MQE REFACTOR VALIDATION CHECKS PASSED')


if __name__ == '__main__':
    main()
