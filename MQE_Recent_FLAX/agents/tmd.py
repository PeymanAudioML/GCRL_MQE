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
from utils.networks import (
    DiscreteStateActionRepresentation,
    GCActor,
    GCDiscreteActor,
    Param,
    StateRepresentation,
)


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=['rng', 'network'],
    meta_fields=['config'],
)
@dataclasses.dataclass(frozen=True)
class TMDAgent:
    """Temporal Metric Distillation (TMD) agent."""

    rng: Any
    network: Any
    config: Any = None

    def replace(self, **kwargs) -> 'TMDAgent':
        return dataclasses.replace(self, **kwargs)

    @jax.jit
    def mrn_distance(self, x, y):
        """Compute the MRN distance between two sets of representations."""
        K = self.config['components']
        assert x.shape[-1] % K == 0

        @jax.jit
        def mrn_distance_component(x, y):
            eps = 1e-6
            d = x.shape[-1]
            mask = jnp.arange(d) < d // 2
            max_component = jnp.max(jax.nn.relu((x - y) * mask), axis=-1)
            l2_component = jnp.sqrt(jnp.square((x - y) * (1 - mask)).sum(axis=-1) + eps)
            assert max_component.shape == l2_component.shape
            return max_component + l2_component

        x_split = jnp.stack(jnp.split(x, K, axis=-1), axis=-1)
        y_split = jnp.stack(jnp.split(y, K, axis=-1), axis=-1)
        dists = jax.vmap(mrn_distance_component, in_axes=(-1, -1), out_axes=-1)(x_split, y_split)

        return dists.mean(axis=-1)

    def iqe_distance(self, x, y):
        """Compute the IQE distance between two sets of representations."""
        k = self.config['components']
        alpha_raw = self.network.select('alpha_raw')()
        alpha = jax.nn.sigmoid(alpha_raw)
        reshape = (x.shape[-1] // k, k)
        x = jnp.reshape(x, (*x.shape[:-1], *reshape))
        y = jnp.reshape(y, (*y.shape[:-1], *reshape))
        valid = x < y
        D = x.shape[-1]
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        neg_inc_copies = jnp.take_along_axis(valid, ixy % D, axis=-1) * jnp.where(ixy < D, -1, 1)
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = (neg_inp_copies < 0) * (-1.0)
        neg_incf = jnp.concatenate([neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1)
        components = (sxy * neg_incf).sum(-1)
        result = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)
        return result

    @jax.jit
    def distance(self, x, y):
        """Dispatch to iqe_distance or mrn_distance based on config."""
        x, y = jnp.broadcast_arrays(x, y)
        if self.config['use_iqe']:
            return self.iqe_distance(x, y)
        else:
            return self.mrn_distance(x, y)

    @jax.jit
    def contrastive_loss(self, batch, grad_params):
        """Compute the contrastive loss."""
        batch_size = batch['observations'].shape[0]

        phi = self.network.select('phi')(batch['observations'], batch['actions'], params=grad_params)
        psi_s = self.network.select('psi')(batch['observations'], params=grad_params)
        psi_next = self.network.select('psi')(batch['next_observations'], params=grad_params)
        psi_g = self.network.select('psi')(batch['value_goals'], params=grad_params)

        # StateRepresentation with ensemble=True returns shape (2, B, d).
        if len(phi.shape) == 2:  # Non-ensemble
            phi = phi[None, ...]
            psi_s = psi_s[None, ...]
            psi_next = psi_next[None, ...]
            psi_g = psi_g[None, ...]

        dist = self.distance(phi[:, :, None], psi_g[:, None, :])
        logits = -dist / jnp.sqrt(phi.shape[-1])
        # logits.shape is (e, B, B)

        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.softmax_cross_entropy(logits=_logits.T, labels=I),
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)
        action_dist = self.distance(psi_s, phi)

        action_invariance_loss = jnp.mean(action_dist)

        dist_next = self.distance(psi_next[:, :, None], psi_g[:, None, :])

        t = self.config['t']
        gamma = self.config['discount']
        if self.config['stopgrad_psi_backup']:
            dist = self.distance(phi[:, :, None], jax.lax.stop_gradient(psi_g[:, None, :]))
        dist_next = jax.lax.stop_gradient(dist_next)

        delta = dist - dist_next
        mask = delta > t
        delta_clipped = jnp.where(mask, t, delta)
        divergence = jnp.where(mask, delta, gamma * jnp.exp(delta_clipped) - dist)

        dw = self.config['diag_backup']
        divergence = divergence * (1 - dw) + jnp.diagonal(divergence, axis1=1, axis2=2)[..., None] * dw
        backup_loss = jnp.mean(divergence)
        divergence = jnp.clip(divergence, None, self.config['t'])

        critic_loss = (
            contrastive_loss
            + action_invariance_loss
            + self.config['zeta'] * backup_loss
        )

        logits = jnp.mean(logits, axis=0)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return critic_loss, {
            'critic_loss': critic_loss,
            'action_invariance_loss': action_invariance_loss,
            'backup_loss': backup_loss,
            'contrastive_loss': contrastive_loss,
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
            'dist': dist.mean(),
            'biggest_diff_in_dist': jnp.max(dist - dist_next),
        }

    @jax.jit
    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss."""
        # Maximize log Q if actor_log_q is True (which is default).
        if self.config['use_latent']:
            psi_s = self.network.select('psi')(batch['observations'], params=grad_params)
            psi_g = self.network.select('psi')(batch['actor_goals'], params=grad_params)
            # psi returns (2, B, d) when ensemble=True; average across ensemble.
            if len(psi_s.shape) == 3:
                psi_s = jnp.mean(psi_s, axis=0)
                psi_g = jnp.mean(psi_g, axis=0)
            if self.config['freeze_enc_for_actor_grad']:
                psi_s, psi_g = jax.lax.stop_gradient(psi_s), jax.lax.stop_gradient(psi_g)
            dist = self.network.select('actor')(psi_s, psi_g, params=grad_params)
        else:
            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

        phi = self.network.select('phi')(batch['observations'], q_actions)
        psi = self.network.select('psi')(batch['actor_goals'])
        # phi and psi are (2, B, d) when ensemble=True.
        q1, q2 = -self.distance(phi, psi)
        q = jnp.minimum(q1, q2)

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        log_prob = dist.log_prob(batch['actions'])

        bc_loss = -(self.config['alpha'] * log_prob).mean()

        actor_loss = q_loss + bc_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'bc_loss': bc_loss,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        critic_loss, critic_info = self.contrastive_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
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
        if self.config['use_latent']:
            psi_s = self.network.select('psi')(observations)
            psi_g = self.network.select('psi')(goals)
            # psi returns (2, B, d) when ensemble=True; average at inference.
            if len(psi_s.shape) == 2:  # in inference, we don't have batch dimension
                psi_s = jnp.mean(psi_s, axis=0)
                psi_g = jnp.mean(psi_g, axis=0)
            dist = self.network.select('actor')(psi_s, psi_g, temperature=temperature)
        else:
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
        """Create a new agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        rngs = nnx.Rngs(init_rng)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        obs_dim = ex_observations.shape[-1]
        latent_dim = config['latent_dim']

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            obs_shape = ex_observations.shape[1:]
            encoder_factory = encoder_modules[config['encoder']]
            encoders['actor'] = GCEncoder(concat_encoder=encoder_factory(obs_shape, rngs))
            encoders['state'] = encoder_factory(obs_shape, rngs)

        # Compute phi/psi input dims.
        if encoders.get('state') is not None:
            state_enc_out = encoders['state'](ex_observations[:1]).shape[-1]
            phi_in = state_enc_out + action_dim
            psi_in = state_enc_out
        else:
            phi_in = obs_dim + action_dim
            psi_in = obs_dim

        # Determine actor input dim.
        if config['use_latent']:
            # Actor takes (psi_s, psi_g) as input — latent_dim + latent_dim.
            actor_in = latent_dim + latent_dim
        elif encoders.get('actor') is not None:
            actor_in = encoders['actor'](ex_observations[:1], ex_goals[:1]).shape[-1]
        else:
            actor_in = obs_dim + obs_dim

        if config['discrete']:
            phi_def = DiscreteStateActionRepresentation(
                in_features=phi_in,
                hidden_dims=config['value_hidden_dims'],
                latent_dim=latent_dim,
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state'),
                rngs=rngs,
            )
            psi_def = DiscreteStateActionRepresentation(
                in_features=psi_in,
                hidden_dims=config['value_hidden_dims'],
                latent_dim=latent_dim,
                action_dim=action_dim,
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state'),
                rngs=rngs,
            )
            actor_def = GCDiscreteActor(
                in_features=actor_in,
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
                rngs=rngs,
            )
        else:
            phi_def = StateRepresentation(
                in_features=phi_in,
                hidden_dims=config['value_hidden_dims'],
                latent_dim=latent_dim,
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state'),
                rngs=rngs,
            )
            psi_def = StateRepresentation(
                in_features=psi_in,
                hidden_dims=config['value_hidden_dims'],
                latent_dim=latent_dim,
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('state'),
                rngs=rngs,
            )
            actor_def = GCActor(
                in_features=actor_in,
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
                rngs=rngs,
            )

        modules = {
            'actor': actor_def,
            'phi': phi_def,
            'psi': psi_def,
        }
        if config['use_iqe']:
            modules['alpha_raw'] = Param(init_value=0.0, rngs=rngs)

        network_def = ModuleDict(modules)
        network_tx = optax.adam(learning_rate=config['lr'])
        network = TrainState.create(network_def, tx=network_tx)

        return cls(rng=rng, network=network, config=dict(config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='tmd',  # Agent name.
            lr=3e-4,
            components=8,  # Number of components in MRN.
            batch_size=512,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            zeta=0.2,
            t=5.0,
            diag_backup=0.5,  # Whether to use diagonal backup.
            stopgrad_psi_backup=True,
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            use_iqe=False,  # Whether to use IQE distance or MRN distance
            use_latent=False,  # Whether to use latent for policy action sampling
            freeze_enc_for_actor_grad=False,  # Whether to stop grad for actor when using encoder
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
