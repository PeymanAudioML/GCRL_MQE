import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax import nnx
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, Sequential, TrainState
from utils.networks import GCActor, GCDiscreteActor, GCValue, Identity, LengthNormalize, MLP


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=['rng', 'network'],
    meta_fields=['config'],
)
@dataclasses.dataclass(frozen=True)
class HIQLAgent:
    """Hierarchical implicit Q-learning (HIQL) agent."""

    rng: Any
    network: Any
    config: Any = None

    def replace(self, **kwargs) -> 'HIQLAgent':
        return dataclasses.replace(self, **kwargs)

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], batch['value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], batch['value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def low_actor_loss(self, batch, grad_params):
        """Compute the low-level actor loss."""
        v1, v2 = self.network.select('value')(batch['observations'], batch['low_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['next_observations'], batch['low_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        # Compute the goal representations of the subgoals.
        goal_reps = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['low_actor_goals']], axis=-1),
            params=grad_params,
        )
        if not self.config['low_actor_rep_grad']:
            # Stop gradients through the goal representations.
            goal_reps = jax.lax.stop_gradient(goal_reps)
        dist = self.network.select('low_actor')(batch['observations'], goal_reps, goal_encoded=True, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
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

    def high_actor_loss(self, batch, grad_params):
        """Compute the high-level actor loss."""
        v1, v2 = self.network.select('value')(batch['observations'], batch['high_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['high_actor_targets'], batch['high_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('high_actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)
        target = self.network.select('goal_rep')(
            jnp.concatenate([batch['observations'], batch['high_actor_targets']], axis=-1)
        )
        log_prob = dist.log_prob(target)

        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - target) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        loss = value_loss + low_actor_loss + high_actor_loss
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
        self.target_update(new_network, 'value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor.

        It first queries the high-level actor to obtain subgoal representations, and then queries the low-level actor
        to obtain raw actions.
        """
        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)
        goal_reps = high_dist.sample(seed=high_seed)
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])

        low_dist = self.network.select('low_actor')(observations, goal_reps, goal_encoded=True, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)

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
        goal_rep_in_dim = obs_dim + obs_dim  # [obs; goal] concatenated

        # Define (state-dependent) subgoal representation phi([s; g]) that outputs a length-normalized vector.
        # IMPORTANT: each use of goal_rep creates SEPARATE instances (no shared parameters).
        if config['encoder'] is not None:
            obs_shape = ex_observations.shape[1:]
            encoder_factory = encoder_modules[config['encoder']]

            # goal_rep: encoder + MLP + LengthNormalize — standalone instance.
            enc_standalone = encoder_factory(obs_shape, rngs)
            enc_out_dim = enc_standalone(ex_observations[:1]).shape[-1]
            goal_rep_def = Sequential([
                enc_standalone,
                MLP(enc_out_dim, (*config['value_hidden_dims'], config['rep_dim']),
                    activate_final=False, layer_norm=config['layer_norm'], rngs=rngs),
                LengthNormalize(rngs=rngs),
            ])

            # Separate encoder instances for value, target_value, low_actor encoders.
            enc_value = encoder_factory(obs_shape, rngs)
            enc_target_value = encoder_factory(obs_shape, rngs)
            enc_low_actor = encoder_factory(obs_shape, rngs)
            enc_high_actor = encoder_factory(obs_shape, rngs)

            # goal_rep instances for value, target_value, low_actor (each with separate params).
            enc_goal_rep_value = encoder_factory(obs_shape, rngs)
            goal_rep_value = Sequential([
                enc_goal_rep_value,
                MLP(enc_out_dim, (*config['value_hidden_dims'], config['rep_dim']),
                    activate_final=False, layer_norm=config['layer_norm'], rngs=rngs),
                LengthNormalize(rngs=rngs),
            ])
            enc_goal_rep_target_value = encoder_factory(obs_shape, rngs)
            goal_rep_target_value = Sequential([
                enc_goal_rep_target_value,
                MLP(enc_out_dim, (*config['value_hidden_dims'], config['rep_dim']),
                    activate_final=False, layer_norm=config['layer_norm'], rngs=rngs),
                LengthNormalize(rngs=rngs),
            ])
            enc_goal_rep_low_actor = encoder_factory(obs_shape, rngs)
            goal_rep_low_actor = Sequential([
                enc_goal_rep_low_actor,
                MLP(enc_out_dim, (*config['value_hidden_dims'], config['rep_dim']),
                    activate_final=False, layer_norm=config['layer_norm'], rngs=rngs),
                LengthNormalize(rngs=rngs),
            ])

            # Value: V(encoder^V(s), phi([s; g]))
            value_encoder_def = GCEncoder(state_encoder=enc_value, concat_encoder=goal_rep_value)
            target_value_encoder_def = GCEncoder(state_encoder=enc_target_value, concat_encoder=goal_rep_target_value)
            # Low-level actor: pi^l(. | encoder^l(s), phi([s; w]))
            low_actor_encoder_def = GCEncoder(state_encoder=enc_low_actor, concat_encoder=goal_rep_low_actor)
            # High-level actor: pi^h(. | encoder^h([s; g]))
            high_actor_encoder_def = GCEncoder(concat_encoder=enc_high_actor)
        else:
            # State-based environments only use the pre-defined shared encoder for subgoal representations.
            # goal_rep: MLP + LengthNormalize — standalone instance.
            goal_rep_def = Sequential([
                MLP(goal_rep_in_dim, (*config['value_hidden_dims'], config['rep_dim']),
                    activate_final=False, layer_norm=config['layer_norm'], rngs=rngs),
                LengthNormalize(rngs=rngs),
            ])

            # Separate goal_rep instances for value, target_value, low_actor.
            goal_rep_value = Sequential([
                MLP(goal_rep_in_dim, (*config['value_hidden_dims'], config['rep_dim']),
                    activate_final=False, layer_norm=config['layer_norm'], rngs=rngs),
                LengthNormalize(rngs=rngs),
            ])
            goal_rep_target_value = Sequential([
                MLP(goal_rep_in_dim, (*config['value_hidden_dims'], config['rep_dim']),
                    activate_final=False, layer_norm=config['layer_norm'], rngs=rngs),
                LengthNormalize(rngs=rngs),
            ])
            goal_rep_low_actor = Sequential([
                MLP(goal_rep_in_dim, (*config['value_hidden_dims'], config['rep_dim']),
                    activate_final=False, layer_norm=config['layer_norm'], rngs=rngs),
                LengthNormalize(rngs=rngs),
            ])

            # Value: V(s, phi([s; g]))
            value_encoder_def = GCEncoder(state_encoder=Identity(rngs=rngs), concat_encoder=goal_rep_value)
            target_value_encoder_def = GCEncoder(state_encoder=Identity(rngs=rngs), concat_encoder=goal_rep_target_value)
            # Low-level actor: pi^l(. | s, phi([s; w]))
            low_actor_encoder_def = GCEncoder(state_encoder=Identity(rngs=rngs), concat_encoder=goal_rep_low_actor)
            # High-level actor: pi^h(. | s, g) (i.e., no encoder)
            high_actor_encoder_def = None

        # Compute value encoder output dimension.
        dummy_value_enc = value_encoder_def(ex_observations[:1], ex_goals[:1])
        value_enc_out = dummy_value_enc.shape[-1]

        # Compute low actor encoder output dimension.
        # For goal_encoded=True: state_encoder(obs) + goal_rep (shape rep_dim).
        dummy_low_enc = low_actor_encoder_def(ex_observations[:1], ex_goals[:1])
        low_actor_enc_out = dummy_low_enc.shape[-1]

        # High actor input dim.
        if high_actor_encoder_def is not None:
            dummy_high_enc = high_actor_encoder_def(ex_observations[:1], ex_goals[:1])
            high_actor_in = dummy_high_enc.shape[-1]
        else:
            high_actor_in = obs_dim + obs_dim

        # Define value and actor networks.
        value_def = GCValue(
            in_features=value_enc_out,
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=value_encoder_def,
            rngs=rngs,
        )
        target_value_def = GCValue(
            in_features=value_enc_out,
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_value_encoder_def,
            rngs=rngs,
        )

        if config['discrete']:
            low_actor_def = GCDiscreteActor(
                in_features=low_actor_enc_out,
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=low_actor_encoder_def,
                rngs=rngs,
            )
        else:
            low_actor_def = GCActor(
                in_features=low_actor_enc_out,
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=low_actor_encoder_def,
                rngs=rngs,
            )

        high_actor_def = GCActor(
            in_features=high_actor_in,
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['rep_dim'],
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=high_actor_encoder_def,
            rngs=rngs,
        )

        network_def = ModuleDict({
            'goal_rep': goal_rep_def,
            'value': value_def,
            'target_value': target_value_def,
            'low_actor': low_actor_def,
            'high_actor': high_actor_def,
        })
        network_tx = optax.adam(learning_rate=config['lr'])
        network = TrainState.create(network_def, tx=network_tx)

        # Initialize target value with same params as value.
        network.params['modules_target_value'] = network.params['modules_value']

        return cls(rng=rng, network=network, config=dict(config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='hiql',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.7,  # IQL expectile.
            low_alpha=3.0,  # Low-level AWR temperature.
            high_alpha=3.0,  # High-level AWR temperature.
            subgoal_steps=25,  # Subgoal steps.
            rep_dim=10,  # Goal representation dimension.
            low_actor_rep_grad=False,  # Whether low-actor gradients flow to goal representation (use True for pixels).
            const_std=True,  # Whether to use constant standard deviation for the actors.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='HGCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
