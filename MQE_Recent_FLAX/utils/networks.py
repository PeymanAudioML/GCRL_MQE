from typing import Any, Optional, Sequence

import chex
import distrax
import jax
import jax.numpy as jnp
from flax import nnx


def default_init(scale: float = 1.0):
    """Default kernel initializer."""
    return nnx.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class Identity(nnx.Module):
    """Identity layer."""

    def __init__(self, *, rngs: nnx.Rngs = None) -> None:
        pass

    def __call__(self, x):
        return x


class MLP(nnx.Module):
    """Multi-layer perceptron.

    Args:
        in_features: Input feature dimension.
        hidden_dims: Hidden layer dimensions (last entry is output dim).
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
        rngs: NNX RNG state.

    Example::

        import jax.numpy as jnp
        from flax import nnx
        mlp = MLP(8, (64, 64), rngs=nnx.Rngs(0))
        y = mlp(jnp.ones((2, 8)))  # shape: (2, 64)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        activations: Any = jax.nn.gelu,
        activate_final: bool = False,
        kernel_init: Any = None,
        layer_norm: bool = False,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        if kernel_init is None:
            kernel_init = default_init()
        self.hidden_dims = tuple(hidden_dims)
        self.activations = activations
        self.activate_final = activate_final
        self.layer_norm = layer_norm

        layers = []
        current_dim = in_features
        for size in hidden_dims:
            layers.append(nnx.Linear(current_dim, size, kernel_init=kernel_init, rngs=rngs))
            current_dim = size
        self.layers = layers

        if layer_norm:
            self.norms = [nnx.LayerNorm(size, rngs=rngs) for size in hidden_dims]

    def __call__(self, x):
        for i, linear in enumerate(self.layers):
            x = linear(x)
            if i + 1 < len(self.layers) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = self.norms[i](x)
        return x


class LengthNormalize(nnx.Module):
    """Length normalization layer.

    Normalizes the input along the last dimension to have a length of
    ``sqrt(dim)``.
    """

    def __init__(self, *, rngs: nnx.Rngs = None) -> None:
        pass

    def __call__(self, x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])


class Param(nnx.Module):
    """Scalar parameter module.

    Args:
        init_value: Initial value of the parameter.
        rngs: NNX RNG state.

    Example::

        from flax import nnx
        p = Param(init_value=0.5, rngs=nnx.Rngs(0))
        v = p()  # scalar jnp.array
    """

    def __init__(self, init_value: float = 0.0, *, rngs: nnx.Rngs = None) -> None:
        self.value = nnx.Param(jnp.full((), init_value))

    def __call__(self):
        return self.value.value


class LogParam(nnx.Module):
    """Scalar parameter module with log scale.

    Args:
        init_value: Initial value (before taking log).
        rngs: NNX RNG state.

    Example::

        from flax import nnx
        lp = LogParam(init_value=1.0, rngs=nnx.Rngs(0))
        alpha = lp()  # scalar jnp.array ≈ 1.0
    """

    def __init__(self, init_value: float = 1.0, *, rngs: nnx.Rngs = None) -> None:
        self.log_value = nnx.Param(jnp.full((), jnp.log(init_value)))

    def __call__(self):
        return jnp.exp(self.log_value.value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


@chex.dataclass(frozen=True)
class RunningMeanStd:
    """Running mean and standard deviation.

    Attributes:
        eps: Epsilon value to avoid division by zero.
        mean: Running mean.
        var: Running variance.
        clip_max: Clip value after normalization.
        count: Number of samples.
    """

    eps: Any = 1e-6
    mean: Any = 1.0
    var: Any = 1.0
    clip_max: Any = 10.0
    count: int = 0

    def normalize(self, batch):
        batch = (batch - self.mean) / jnp.sqrt(self.var + self.eps)
        batch = jnp.clip(batch, -self.clip_max, self.clip_max)
        return batch

    def unnormalize(self, batch):
        return batch * jnp.sqrt(self.var + self.eps) + self.mean

    def update(self, batch):
        batch_mean, batch_var = jnp.mean(batch, axis=0), jnp.var(batch, axis=0)
        batch_count = len(batch)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        return self.replace(mean=new_mean, var=new_var, count=total_count)


class GCActor(nnx.Module):
    """Goal-conditioned actor.

    Args:
        in_features: Input feature dimension (obs_dim [+ goal_dim] or encoder output dim).
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
        rngs: NNX RNG state.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        action_dim: int,
        log_std_min: Optional[float] = -5,
        log_std_max: Optional[float] = 2,
        tanh_squash: bool = False,
        state_dependent_std: bool = False,
        const_std: bool = True,
        final_fc_init_scale: float = 1e-2,
        gc_encoder=None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.tanh_squash = tanh_squash
        self.state_dependent_std = state_dependent_std
        self.const_std = const_std
        self.gc_encoder = gc_encoder

        # If gc_encoder is provided, its output feeds into actor_net.
        # We can't know gc_encoder output dim statically here, so in_features
        # must be set by caller to the encoder output dim when gc_encoder != None.
        self.actor_net = MLP(in_features, hidden_dims, activate_final=True, rngs=rngs)
        actor_out_dim = hidden_dims[-1]
        self.mean_net = nnx.Linear(
            actor_out_dim, action_dim,
            kernel_init=default_init(final_fc_init_scale),
            rngs=rngs,
        )
        if state_dependent_std:
            self.log_std_net = nnx.Linear(
                actor_out_dim, action_dim,
                kernel_init=default_init(final_fc_init_scale),
                rngs=rngs,
            )
        elif not const_std:
            # Learnable but not state-dependent log stds.
            self.log_stds = nnx.Param(jnp.zeros((action_dim,)))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded: bool = False,
        temperature: float = 1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Scaling factor for the standard deviation.
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds.value

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash:
            distribution = TransformedWithMode(
                distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )

        return distribution


class GCDiscreteActor(nnx.Module):
    """Goal-conditioned actor for discrete actions.

    Args:
        in_features: Input feature dimension.
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
        rngs: NNX RNG state.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        action_dim: int,
        final_fc_init_scale: float = 1e-2,
        gc_encoder=None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.gc_encoder = gc_encoder
        self.actor_net = MLP(in_features, hidden_dims, activate_final=True, rngs=rngs)
        actor_out_dim = hidden_dims[-1]
        self.logit_net = nnx.Linear(
            actor_out_dim, action_dim,
            kernel_init=default_init(final_fc_init_scale),
            rngs=rngs,
        )

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded: bool = False,
        temperature: float = 1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Inverse scaling factor for the logits (set to 0 to get the argmax).
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        logits = self.logit_net(outputs)

        distribution = distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))

        return distribution


class GCValue(nnx.Module):
    """Goal-conditioned value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Args:
        in_features: Input feature dimension.
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function (creates 2 MLPs).
        value_exp: Whether to exponentiate the value.
        gc_encoder: Optional GCEncoder module to encode the inputs.
        rngs: NNX RNG state.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        layer_norm: bool = True,
        ensemble: bool = True,
        value_exp: bool = False,
        gc_encoder=None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.ensemble = ensemble
        self.value_exp = value_exp
        self.gc_encoder = gc_encoder

        out_dims = (*hidden_dims, 1)
        if ensemble:
            # Two separate MLPs for ensemble.
            self.value_net1 = MLP(
                in_features, out_dims, activate_final=False, layer_norm=layer_norm, rngs=rngs
            )
            self.value_net2 = MLP(
                in_features, out_dims, activate_final=False, layer_norm=layer_norm, rngs=rngs
            )
        else:
            self.value_net = MLP(
                in_features, out_dims, activate_final=False, layer_norm=layer_norm, rngs=rngs
            )

    def __call__(self, observations, goals=None, actions=None):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
        """
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        if self.ensemble:
            v1 = self.value_net1(inputs).squeeze(-1)
            v2 = self.value_net2(inputs).squeeze(-1)
            v = jnp.stack([v1, v2], axis=0)
        else:
            v = self.value_net(inputs).squeeze(-1)

        if self.value_exp:
            v = jnp.exp(v)

        return v


class GCDiscreteCritic(GCValue):
    """Goal-conditioned critic for discrete actions."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        action_dim: int,
        layer_norm: bool = True,
        ensemble: bool = True,
        value_exp: bool = False,
        gc_encoder=None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(
            in_features, hidden_dims,
            layer_norm=layer_norm, ensemble=ensemble, value_exp=value_exp,
            gc_encoder=gc_encoder, rngs=rngs,
        )
        self.action_dim = action_dim

    def __call__(self, observations, goals=None, actions=None):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions)


class GCBilinearValue(nnx.Module):
    """Goal-conditioned bilinear value/critic function.

    Computes V(s, g) = phi(s)^T psi(g) / sqrt(d) or
    Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d).

    Args:
        in_features: Input feature dimension for the phi network.
        goal_in_features: Input feature dimension for the psi network.
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        value_exp: Whether to exponentiate the value.
        state_encoder: Optional state encoder.
        goal_encoder: Optional goal encoder.
        rngs: NNX RNG state.
    """

    def __init__(
        self,
        in_features: int,
        goal_in_features: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        layer_norm: bool = True,
        ensemble: bool = True,
        value_exp: bool = False,
        state_encoder=None,
        goal_encoder=None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.latent_dim = latent_dim
        self.ensemble = ensemble
        self.value_exp = value_exp
        self.state_encoder = state_encoder
        self.goal_encoder = goal_encoder

        out_dims = (*hidden_dims, latent_dim)
        if ensemble:
            self.phi1 = MLP(in_features, out_dims, activate_final=False, layer_norm=layer_norm, rngs=rngs)
            self.phi2 = MLP(in_features, out_dims, activate_final=False, layer_norm=layer_norm, rngs=rngs)
            self.psi1 = MLP(goal_in_features, out_dims, activate_final=False, layer_norm=layer_norm, rngs=rngs)
            self.psi2 = MLP(goal_in_features, out_dims, activate_final=False, layer_norm=layer_norm, rngs=rngs)
        else:
            self.phi = MLP(in_features, out_dims, activate_final=False, layer_norm=layer_norm, rngs=rngs)
            self.psi = MLP(goal_in_features, out_dims, activate_final=False, layer_norm=layer_norm, rngs=rngs)

    def __call__(self, observations, goals, actions=None, info: bool = False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions (optional).
            info: Whether to additionally return the representations phi and psi.
        """
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        if self.ensemble:
            phi1 = self.phi1(phi_inputs)
            phi2 = self.phi2(phi_inputs)
            psi1 = self.psi1(goals)
            psi2 = self.psi2(goals)
            phi = jnp.stack([phi1, phi2], axis=0)  # (2, B, d)
            psi = jnp.stack([psi1, psi2], axis=0)  # (2, B, d)
            v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)  # (2, B)
        else:
            phi = self.phi(phi_inputs)
            psi = self.psi(goals)
            v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi, psi
        else:
            return v


class GCDiscreteBilinearCritic(GCBilinearValue):
    """Goal-conditioned bilinear critic for discrete actions."""

    def __init__(
        self,
        in_features: int,
        goal_in_features: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        action_dim: int,
        layer_norm: bool = True,
        ensemble: bool = True,
        value_exp: bool = False,
        state_encoder=None,
        goal_encoder=None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(
            in_features, goal_in_features, hidden_dims, latent_dim,
            layer_norm=layer_norm, ensemble=ensemble, value_exp=value_exp,
            state_encoder=state_encoder, goal_encoder=goal_encoder, rngs=rngs,
        )
        self.action_dim = action_dim

    def __call__(self, observations, goals=None, actions=None, info: bool = False):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions, info)


class GCMRNValue(nnx.Module):
    """Metric residual network (MRN) value function.

    Computes V(s, g) as the sum of a symmetric Euclidean distance and an
    asymmetric L^infinity-based quasimetric.

    Args:
        in_features: Input feature dimension.
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        value_exp: Whether to exponentiate the value.
        encoder: Optional state/goal encoder.
        rngs: NNX RNG state.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        layer_norm: bool = True,
        value_exp: bool = False,
        encoder=None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.latent_dim = latent_dim
        self.value_exp = value_exp
        self.encoder = encoder
        self.phi = MLP(
            in_features, (*hidden_dims, latent_dim),
            activate_final=False, layer_norm=layer_norm, rngs=rngs,
        )

    def __call__(self, observations, goals, is_phi: bool = False, info: bool = False):
        """Return the MRN value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        sym_s = phi_s[..., : self.latent_dim // 2]
        sym_g = phi_g[..., : self.latent_dim // 2]
        asym_s = phi_s[..., self.latent_dim // 2 :]
        asym_g = phi_g[..., self.latent_dim // 2 :]
        squared_dist = ((sym_s - sym_g) ** 2).sum(axis=-1)
        quasi = jax.nn.relu((asym_s - asym_g).max(axis=-1))
        v = jnp.sqrt(jnp.maximum(squared_dist, 1e-12)) + quasi

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi_s, phi_g
        else:
            return v


class GCIQEValue(nnx.Module):
    """Interval quasimetric embedding (IQE) value function.

    Computes the value function as an IQE-based quasimetric.

    Args:
        in_features: Input feature dimension.
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        dim_per_component: Dimension of each component in IQE.
        layer_norm: Whether to apply layer normalization.
        value_exp: Whether to exponentiate the value.
        encoder: Optional state/goal encoder.
        rngs: NNX RNG state.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        dim_per_component: int,
        layer_norm: bool = True,
        value_exp: bool = False,
        encoder=None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.latent_dim = latent_dim
        self.dim_per_component = dim_per_component
        self.value_exp = value_exp
        self.encoder = encoder
        self.phi = MLP(
            in_features, (*hidden_dims, latent_dim),
            activate_final=False, layer_norm=layer_norm, rngs=rngs,
        )
        # alpha: scalar blending weight between mean and max of IQE components.
        self.alpha = Param(init_value=0.0, rngs=rngs)

    def __call__(self, observations, goals, is_phi: bool = False, info: bool = False):
        """Return the IQE value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        alpha = jax.nn.sigmoid(self.alpha())
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        x = jnp.reshape(phi_s, (*phi_s.shape[:-1], -1, self.dim_per_component))
        y = jnp.reshape(phi_g, (*phi_g.shape[:-1], -1, self.dim_per_component))
        valid = x < y
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        neg_inc_copies = jnp.take_along_axis(valid, ixy % self.dim_per_component, axis=-1) * jnp.where(
            ixy < self.dim_per_component, -1, 1
        )
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = -1.0 * (neg_inp_copies < 0)
        neg_incf = jnp.concatenate([neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1)
        components = (sxy * neg_incf).sum(axis=-1)
        v = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi_s, phi_g
        else:
            return v


class StateRepresentation(nnx.Module):
    """State representation module.

    Args:
        in_features: Input feature dimension.
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        value_exp: Whether to exponentiate the value.
        state_encoder: Optional state encoder.
        rngs: NNX RNG state.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        layer_norm: bool = True,
        ensemble: bool = True,
        value_exp: bool = False,
        state_encoder=None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.ensemble = ensemble
        self.value_exp = value_exp
        self.state_encoder = state_encoder

        out_dims = (*hidden_dims, latent_dim)
        if ensemble:
            self.phi1 = MLP(in_features, out_dims, activate_final=False, layer_norm=layer_norm, rngs=rngs)
            self.phi2 = MLP(in_features, out_dims, activate_final=False, layer_norm=layer_norm, rngs=rngs)
        else:
            self.phi = MLP(in_features, out_dims, activate_final=False, layer_norm=layer_norm, rngs=rngs)

    def __call__(self, observations, actions=None, info: bool = False):
        """Return the state representation.

        Args:
            observations: Observations.
            actions: Actions (optional).
            info: Unused; kept for API compatibility.
        """
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)

        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        if self.ensemble:
            phi1 = self.phi1(phi_inputs)
            phi2 = self.phi2(phi_inputs)
            phi = jnp.stack([phi1, phi2], axis=0)  # (2, B, latent_dim)
        else:
            phi = self.phi(phi_inputs)

        if self.value_exp:
            phi = jnp.exp(phi)

        return phi


class DiscreteStateActionRepresentation(StateRepresentation):
    """State representation module for discrete actions."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        action_dim: int,
        layer_norm: bool = True,
        ensemble: bool = True,
        value_exp: bool = False,
        state_encoder=None,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__(
            in_features, hidden_dims, latent_dim,
            layer_norm=layer_norm, ensemble=ensemble, value_exp=value_exp,
            state_encoder=state_encoder, rngs=rngs,
        )
        self.action_dim = action_dim

    def __call__(self, observations, actions=None, info: bool = False):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)

        if actions is not None:
            actions = jnp.eye(self.action_dim)[actions]

        return super().__call__(observations, actions, info)
