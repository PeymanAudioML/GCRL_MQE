import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax import nnx
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import GCActor, GCDiscreteActor


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=['rng', 'network'],
    meta_fields=['config'],
)
@dataclasses.dataclass(frozen=True)
class GCBCAgent:
    """Goal-conditioned behavioral cloning (GCBC) agent."""

    rng: Any
    network: Any
    config: Any = None

    def replace(self, **kwargs) -> 'GCBCAgent':
        return dataclasses.replace(self, **kwargs)

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the BC actor loss."""
        dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -log_prob.mean()

        actor_info = {
            'actor_loss': actor_loss,
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            actor_info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

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
        if not self.config['discrete']:
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
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        rngs = nnx.Rngs(init_rng)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        obs_dim = ex_observations.shape[-1]

        # Define encoder.
        encoders = dict()
        if config['encoder'] is not None:
            obs_shape = ex_observations.shape[1:]  # (H, W, C) for pixel obs
            encoder_factory = encoder_modules[config['encoder']]
            encoders['actor'] = GCEncoder(
                concat_encoder=encoder_factory(obs_shape, rngs)
            )

        # Determine actor input dimension.
        if encoders.get('actor') is not None:
            # Encoder output dim is determined by the mlp_hidden_dims[-1] of ImpalaEncoder.
            # We compute it by running a dummy forward pass.
            dummy_enc_out = encoders['actor'](ex_observations[:1], ex_goals[:1])
            actor_in_features = dummy_enc_out.shape[-1]
        else:
            actor_in_features = obs_dim + obs_dim  # obs + goal

        # Define actor network.
        if config['discrete']:
            actor_def = GCDiscreteActor(
                in_features=actor_in_features,
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
                rngs=rngs,
            )
        else:
            actor_def = GCActor(
                in_features=actor_in_features,
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
                rngs=rngs,
            )

        network_def = ModuleDict({'actor': actor_def})
        network_tx = optax.adam(learning_rate=config['lr'])
        network = TrainState.create(network_def, tx=network_tx)

        return cls(rng=rng, network=network, config=dict(config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='gcbc',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            discount=0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_p_trajgoal=1.0,  # Unused (defined for compatibility with GCDataset).
            value_p_randomgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_geom_sample=False,  # Unused (defined for compatibility with GCDataset).
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
