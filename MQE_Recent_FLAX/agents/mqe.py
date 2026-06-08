from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax import nnx
from utils.encoders import GCEncoder, encoder_modules
from utils.networks import (
    DiscreteStateActionRepresentation,
    GCActor,
    GCDiscreteActor,
    StateRepresentation,
)


class MQENetwork(nnx.Module):
    """Native flax.nnx container holding the MQE sub-networks.

    Stores the three modules the MQE algorithm uses:
    - ``actor``: the (goal-conditioned) policy network.
    - ``phi``:   the state-action representation phi(s, a).
    - ``psi``:   the state/goal representation psi(s) / psi(g).

    ``select`` is kept for parity with the previous ModuleDict interface, so
    losses can be written as either ``model.phi(...)`` or
    ``model.select('phi')(...)``.
    """

    def __init__(self, actor, phi, psi) -> None:
        self.actor = actor
        self.phi = phi
        self.psi = psi

    def select(self, name: str):
        return getattr(self, name)


@nnx.jit
def _mqe_sample_actions(model, observations, goals, seed, temperature):
    """JIT-compiled action sampling from the MQE actor."""
    dist = model.actor(observations, goals, temperature=temperature)
    return dist.sample(seed=seed)


def _build_update_step(agent):
    """Build the per-agent, nnx-native jitted training step.

    The returned function takes the live ``model`` and ``optimizer`` graph
    nodes (plus ``rng`` and ``batch``) and mutates them in place across the
    ``nnx.jit`` boundary, exactly like the standard flax.nnx train-step
    pattern. ``agent`` is captured only for its (static) ``config`` and its
    loss methods; the jitted region never reads ``agent.model`` /
    ``agent.optimizer`` (it uses the passed-in args), so there is no aliasing.
    """

    @nnx.jit
    def _update_step(model, optimizer, rng, batch):
        def loss_fn(m):
            return agent.total_loss(m, batch, rng)

        (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

        # Gradient statistics, preserved from the old apply_loss_fn().
        grad_max = jax.tree_util.tree_map(jnp.max, grads)
        grad_min = jax.tree_util.tree_map(jnp.min, grads)
        grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)
        grad_max_flat = jnp.concatenate(
            [jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0
        )
        grad_min_flat = jnp.concatenate(
            [jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0
        )
        grad_norm_flat = jnp.concatenate(
            [jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0
        )

        info = dict(info)
        info['grad/max'] = jnp.max(grad_max_flat)
        info['grad/min'] = jnp.min(grad_min_flat)
        info['grad/norm'] = jnp.linalg.norm(grad_norm_flat, ord=1)

        # In-place parameter update (mutates `model`, which the optimizer holds).
        optimizer.update(grads)
        return info

    return _update_step


class MQEAgent:
    """Multistep Quasimetric Estimation (MQE) agent.

    This MQE implementation uses native flax.nnx modules and nnx.Optimizer
    instead of the older custom TrainState split/merge wrapper. The agent is a
    plain (mutable) Python object holding:

        MQEAgent
        ├── rng        (jax PRNGKey, advanced on every update)
        ├── model      (MQENetwork: actor / phi / psi)
        ├── optimizer  (nnx.Optimizer, wrt=nnx.Param)
        └── config     (plain dict of hyperparameters)

    Training flow (see ``update``):

        batch = train_dataset.sample(batch_size)
        loss, grads = nnx.value_and_grad(total_loss)(model)
        optimizer.update(grads)        # mutates model in place

    The MQE algorithm, loss equations, RNG handling, and logged metrics are
    unchanged from the TrainState-based implementation.
    """

    def __init__(self, rng, model, optimizer, config) -> None:
        self.rng = rng
        self.model = model
        self.optimizer = optimizer
        self.config = config
        # Lazily-built jitted step; set by create(). Eval-only copies
        # (see to_device) leave this as None since they never update.
        self._update_step = None

    def mrn_distance(self, x: jnp.ndarray, y: jnp.ndarray):
        """Compute the MRN distance between two sets of representations."""
        K = self.config['components']
        assert x.shape[-1] % K == 0

        def mrn_distance_component(x: jnp.ndarray, y: jnp.ndarray):
            eps = 1e-8
            d = x.shape[-1]
            mask = jnp.arange(d) < d // 2
            max_component: jnp.ndarray = jax.nn.relu(jnp.max((x - y) * mask, axis=-1))
            l2_component: jnp.ndarray = jnp.linalg.norm((x - y) * (1 - mask) + eps, axis=-1)
            return max_component + l2_component

        x_split = jnp.stack(jnp.split(x, K, axis=-1), axis=-1)
        y_split = jnp.stack(jnp.split(y, K, axis=-1), axis=-1)
        dists: jnp.ndarray = jax.vmap(mrn_distance_component, in_axes=(-1, -1), out_axes=-1)(x_split, y_split)

        return dists.mean(axis=-1) / jnp.sqrt(x.shape[-1])

    def distance(self, x, y) -> jnp.ndarray:
        """Compute the distance between two sets of representations."""
        return self.mrn_distance(x, y)

    def critic_loss(self, model, batch, critic_rng):
        """Compute the MQE critic loss."""
        batch_size = self.config['batch_size']
        key = jax.random.PRNGKey(critic_rng[1])
        use_next_state = jax.random.bernoulli(key, p=self.config['next_state_sample'], shape=(batch_size,))
        use_next_state_mask = jnp.reshape(
            use_next_state, (batch_size, *[1] * (len(batch['observations'].shape) - 1))
        )
        intermediate_value_goals = jnp.where(
            use_next_state_mask, batch['next_observations'], batch['intermediate_value_goals']
        )

        batch_size = batch['observations'].shape[0]
        phi = model.phi(batch['observations'], batch['actions'])
        psi_s = model.psi(batch['observations'])
        psi_next = model.psi(intermediate_value_goals)
        psi_g = model.psi(batch['value_goals'])

        # StateRepresentation with ensemble=True returns (2, B, d).
        if len(psi_s.shape) == 2:  # Non-ensemble
            phi = phi[None, ...]
            psi_s = psi_s[None, ...]
            psi_next = psi_next[None, ...]
            psi_g = psi_g[None, ...]

        dist = self.distance(phi[:, :, None], psi_g[:, None, :])
        dist_next = self.distance(psi_next[:, :, None], psi_g[:, None, :])

        I = jnp.eye(batch_size)
        logits = -dist

        action_dist = self.distance(psi_s, phi)
        action_invariance_loss = jnp.mean(jnp.square(jnp.exp(-action_dist) - 1))

        def compute_backup(dist, dist_next):
            t = self.config['t']
            gamma = self.config['discount']
            delta = dist - dist_next
            mask = delta > t
            delta_clipped = jnp.where(mask, t, delta)
            one_step_mask = jnp.where(
                use_next_state_mask.reshape(use_next_state_mask.shape[0],),
                1.0,
                batch['intermediate_value_goals_offsets'],
            )[None, :, None]

            s = gamma ** one_step_mask
            divergence = jnp.where(mask, delta, s * jnp.exp(delta_clipped) - dist)
            dw = self.config['diag_backup']
            optim_value = 1 - jax.lax.stop_gradient(dist_next) + jnp.log(gamma) * one_step_mask
            optim_value = optim_value * (1 - dw) + jnp.diagonal(optim_value, axis1=1, axis2=2)[..., None] * dw
            diag = jnp.diagonal(divergence, axis1=1, axis2=2)[..., None] * dw
            divergence = divergence * (1 - dw) + diag
            optim_backup = jnp.mean(optim_value)
            return jnp.mean(divergence), optim_backup

        # optim_backup=0 -> recovers behavior distance
        backup_loss, optim_backup = compute_backup(dist, jax.lax.stop_gradient(dist_next))
        optim_backup = jnp.mean(optim_backup)

        critic_loss = backup_loss + action_invariance_loss
        logits = jnp.mean(logits, axis=0)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return (
            critic_loss,
            {
                'critic_loss': critic_loss,
                'backup_loss': backup_loss,
                'backup_optim_loss': backup_loss - optim_backup,
                'action_invariance_loss': action_invariance_loss,
                'binary_accuracy': jnp.mean((logits > 0) == I),
                'categorical_accuracy': jnp.mean(correct),
                'logits_pos': logits_pos,
                'logits_neg': logits_neg,
                'logits': logits.mean(),
                'dist': dist.mean(),
                # debug metrics
                'phi_mag': jnp.mean(jnp.abs(phi)),
                'psi_s_mag': jnp.mean(jnp.abs(psi_s)),
                'biggest_diff_in_dist': jnp.max(dist - dist_next),
            },
        )

    def actor_loss(self, model, batch, rng=None):
        """Compute the DDPG+BC actor loss."""
        # Maximize log Q if actor_log_q is True (which is default).
        dist = model.actor(batch['observations'], batch['actor_goals'])
        if self.config['const_std']:
            q_actions = jnp.clip(dist.mode(), -1, 1)
        else:
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
        phi = model.phi(batch['observations'], q_actions)
        psi_g = model.psi(batch['actor_goals'])
        # phi and psi_g are (2, B, d) when ensemble=True.
        q1, q2 = -self.distance(phi, psi_g)
        q = jnp.minimum(q1, q2)

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        if self.config["normalize_q_loss"]:
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
        else:
            q_loss = -q.mean()
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

    def total_loss(self, model, batch, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, critic_rng = jax.random.split(rng)

        critic_loss, critic_info = self.critic_loss(model, batch, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(model, batch, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        total_loss = critic_loss + actor_loss
        return total_loss, info

    def update(self, batch):
        """Update the agent in place and return ``(self, info)``.

        Uses the nnx-native pattern: gradients are taken w.r.t. the model's
        nnx.Param state and applied via ``nnx.Optimizer.update`` inside an
        ``nnx.jit`` step that mutates ``self.model`` and ``self.optimizer`` in
        place. ``self`` is returned for compatibility with the previous
        ``agent, info = agent.update(batch)`` call site.
        """
        new_rng, rng = jax.random.split(self.rng)
        info = self._update_step(self.model, self.optimizer, rng, batch)
        self.rng = new_rng
        return self, info

    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        actions = _mqe_sample_actions(
            self.model, observations, goals, seed, jnp.asarray(temperature, dtype=jnp.float32)
        )
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    def get_distance(self, observations, goals, actions):
        """Compute distance Q(s,a,g) or V(s,g) depending on config."""
        # whether want to compute the action-conditioned distance d(phi(s,a), psi(g)) aka Q(s, a, g)
        # or action-free distance d(phi(s), psi(g)) aka V(s, g)
        if self.config['use_action_for_distance']:
            phi = self.model.phi(observations, actions)
        else:
            phi = self.model.psi(observations)
        psi = self.model.psi(goals)
        return self.distance(phi, psi)

    def to_device(self, device):
        """Return an eval-only copy of the agent with its model on ``device``.

        Replaces the old ``jax.device_put(agent, ...)`` path (NNX modules are
        not directly device_put-able). The returned agent shares ``config`` and
        carries only what evaluation needs (``model`` + ``rng``); it has no
        optimizer or update step.
        """
        graphdef, state = nnx.split(self.model)
        state = jax.device_put(state, device)
        model = nnx.merge(graphdef, state)
        return MQEAgent(
            rng=jax.device_put(self.rng, device),
            model=model,
            optimizer=None,
            config=self.config,
        )

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        This MQE implementation uses native flax.nnx modules and nnx.Optimizer
        instead of the older custom TrainState split/merge wrapper.
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
        latent_dim = config['latent_dim']

        config['gamma'] = config['discount']

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            obs_shape = ex_observations.shape[1:]
            encoder_factory = encoder_modules[config['encoder']]
            if not config['use_latent']:
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
        if encoders.get('actor') is not None:
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

        # Native flax.nnx model + optimizer (replaces ModuleDict + TrainState).
        model = MQENetwork(actor=actor_def, phi=phi_def, psi=psi_def)
        tx = optax.adam(learning_rate=config['lr'])
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        agent = cls(rng=rng, model=model, optimizer=optimizer, config=dict(config))
        agent._update_step = _build_update_step(agent)
        return agent


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            # Network hyperparameters.
            agent_name='mqe',  # Agent name.
            lr=3e-4,  # Learning rate.
            components=8,  # Number of components to average in the MRN/IQE distance ensemble.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            latent_dim=512,  # Latent dimension for actors/encoders.
            layer_norm=True,  # Whether to use layer normalization for networks.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            normalize_q_loss=True,  # Whether to normalize Q loss.
            use_latent=False,  # Whether to use latent for policy action sampling.

            # MQE hyperparameters
            discount=0.995,  # Discount factor for sampling value_goal via geometric dist.
            lambda_=0.95,  # lambda for sampling intermediate_value_goal via geometric dist.
            next_state_sample=0.2,  # probability of using next state as intermediate_value_goal.
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            t=5.0,  # Clipping threshold for the backup LINEX loss.
            diag_backup=0.5,  # Weighting of backups on diagonal vs. off-diagonal.

            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric distribution for sampling for future value goals.
            intermediate_value_geom_sample=True,  # Whether to use geometric sampling for intermediate value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused for this method (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation. Unused for state-based methods.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.

            # Toggle plotting
            use_action_for_distance=True,  # Whether to use action for distance computation Q(s, a, g) or V(s, g)
        )
    )
    return config
