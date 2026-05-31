import copy
import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax import nnx
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import GCActor, GCValue, LogParam


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=['rng', 'network'],
    meta_fields=['config'],
)
@dataclasses.dataclass(frozen=True)
class NGCSACBCAgent:
    """n-step goal-conditioned soft actor-critic + behavioral cloning (n-step GCSAC+BC) agent."""

    rng: Any
    network: Any
    config: Any = None

    def replace(self, **kwargs) -> 'NGCSACBCAgent':
        return dataclasses.replace(self, **kwargs)

    def critic_loss(self, batch, grad_params, rng):
        """Compute the n-step SAC critic loss."""
        rng, sample_rng = jax.random.split(rng)
        next_dist = self.network.select('actor')(
            batch['high_value_next_observations'], goals=batch['high_value_goals']
        )
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=sample_rng)

        next_qs = self.network.select('target_critic')(
            batch['high_value_next_observations'], goals=batch['high_value_goals'], actions=next_actions
        )
        if self.config['value_loss_type'] == 'bce':
            next_qs = jax.nn.sigmoid(next_qs)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        elif self.config['q_agg'] == 'mean':
            next_q = next_qs.mean(axis=0)

        target_q = (
            batch['high_value_rewards']
            + (self.config['discount'] ** batch['high_value_subgoal_steps']) * batch['high_value_masks'] * next_q
        )
        # Clip target Q values to the valid range.
        if self.config['gc_negative']:
            target_q = jnp.clip(target_q, -1 / (1 - self.config['discount']), 0)
        else:
            target_q = jnp.clip(target_q, 0, 1)

        if self.config['value_loss_type'] == 'squared':
            q = self.network.select('critic')(
                batch['observations'],
                goals=batch['high_value_goals'],
                actions=batch['actions'],
                params=grad_params,
            )
            critic_loss = jnp.square(q - target_q).mean()

            return critic_loss, {
                'critic_loss': critic_loss,
                'q_mean': q.mean(),
                'q_max': q.max(),
                'q_min': q.min(),
            }
        elif self.config['value_loss_type'] == 'bce':
            q_logit = self.network.select('critic')(
                batch['observations'],
                goals=batch['high_value_goals'],
                actions=batch['actions'],
                params=grad_params,
            )
            q = jax.nn.sigmoid(q_logit)
            log_q = jax.nn.log_sigmoid(q_logit)
            log_not_q = jax.nn.log_sigmoid(-q_logit)
            critic_loss = -(log_q * target_q + log_not_q * (1 - target_q)).mean()

            return critic_loss, {
                'critic_loss': critic_loss,
                'q_mean': q.mean(),
                'q_max': q.max(),
                'q_min': q.min(),
                'q_logit_mean': q_logit.mean(),
                'q_logit_max': q_logit.max(),
                'q_logit_min': q_logit.min(),
            }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the SAC actor loss."""
        # Actor loss.
        dist = self.network.select('actor')(
            batch['observations'], goals=batch['high_actor_goals'], params=grad_params
        )
        actions, log_probs = dist.sample_and_log_prob(seed=rng)

        qs = self.network.select('critic')(
            batch['observations'], goals=batch['high_actor_goals'], actions=actions
        )
        q = jnp.mean(qs, axis=0)

        actor_loss = (log_probs * self.network.select('lam')() - q).mean()

        # Entropy loss.
        lam = self.network.select('lam')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        lam_loss = (lam * (entropy - self.config['target_entropy'])).mean()

        # BC loss.
        mse = jnp.square(actions - batch['actions']).sum(axis=-1)
        bc_loss = (self.config['alpha'] * mse).mean()

        # Make it scale-invariant.
        scaler = jax.lax.stop_gradient(jnp.abs(q).mean())

        total_loss = actor_loss + lam_loss + bc_loss * scaler

        if self.config['tanh_squash']:
            action_std = dist._distribution.stddev()
        else:
            action_std = dist.stddev().mean()

        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'lam_loss': lam_loss,
            'bc_loss': bc_loss,
            'scaled_bc_loss': bc_loss * scaler,
            'lam': lam,
            'entropy': -log_probs.mean(),
            'std': action_std.mean(),
            'mse': mse.mean(),
            'q': q.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            example_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        rngs = nnx.Rngs(init_rng)

        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]
        obs_dim = ex_observations.shape[-1]
        goal_dim = ex_goals.shape[-1]

        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim

        # GCValue critic: called with (obs, goals=goal, actions=action).
        critic_in = obs_dim + goal_dim + action_dim

        # Define networks.
        # ensemble=True when num_qs > 1, else ensemble=False (single MLP).
        use_ensemble = config['num_qs'] > 1
        critic_def = GCValue(
            in_features=critic_in,
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=use_ensemble,
            rngs=rngs,
        )
        target_critic_def = GCValue(
            in_features=critic_in,
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=use_ensemble,
            rngs=rngs,
        )

        # GCActor: called with (obs, goals=goal).
        actor_in = obs_dim + goal_dim
        actor_def = GCActor(
            in_features=actor_in,
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            tanh_squash=config['tanh_squash'],
            state_dependent_std=config['state_dependent_std'],
            const_std=False,
            final_fc_init_scale=config['actor_fc_scale'],
            rngs=rngs,
        )

        # Define the dual lam variable.
        lam_def = LogParam(rngs=rngs)

        network_def = ModuleDict({
            'critic': critic_def,
            'target_critic': target_critic_def,
            'actor': actor_def,
            'lam': lam_def,
        })
        network_tx = optax.adam(learning_rate=config['lr'])
        network = TrainState.create(network_def, tx=network_tx)

        # Initialize target critic with same params as critic.
        network.params['modules_target_critic'] = network.params['modules_critic']

        return cls(rng=rng, network=network, config=dict(config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='ngcsacbc',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.999,  # Discount factor.
            tau=0.005,  # Target network update rate.
            target_entropy=ml_collections.config_dict.placeholder(float),  # Target entropy (None for automatic tuning).
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
            target_entropy_multiplier=0.5,  # Multiplier to dim(A) for target entropy.
            tanh_squash=True,  # Whether to squash actions with tanh.
            state_dependent_std=True,  # Whether to use state-dependent standard deviations for actor.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            num_qs=2,  # Number of Q ensembles.
            q_agg='min',  # Aggregation function for Q values.
            alpha=0.1,  # BC coefficient.
            value_loss_type='bce',  # Value loss type ('squared' or 'bce').
            # Dataset hyperparameters.
            dataset_class='HGCDataset',  # Dataset class name.
            subgoal_steps=50,  # Subgoal steps.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=False,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.5,  # Probability of using a random state as the actor goal.
            actor_geom_sample=True,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            discrete=False,  # Whether the action space is discrete.
            cotrain_steps=1000000,  # Number of steps to cotrain the actor.
        )
    )
    return config
