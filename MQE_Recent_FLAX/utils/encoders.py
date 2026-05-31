import functools
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from utils.networks import MLP


def _max_pool(x: jax.Array, window_shape=(3, 3), strides=(2, 2), padding='SAME') -> jax.Array:
    """Spatial max-pooling over the H and W axes of a 4-D NHWC array.

    This is a free-function replacement for ``flax.linen.max_pool`` that uses
    ``jax.lax.reduce_window`` directly.

    Args:
        x: Input array of shape ``(N, H, W, C)``.
        window_shape: Height and width of the pooling window.
        strides: Height and width strides.
        padding: ``'SAME'`` or ``'VALID'``.

    Returns:
        Pooled array.
    """
    # reduce_window expects padding as a sequence of (low, high) pairs, one per
    # axis.  For batch and channel dims we use no padding.
    if padding == 'SAME':
        # Compute symmetric padding the same way TF/Flax do.
        pad_h = max(window_shape[0] - strides[0], 0)
        pad_w = max(window_shape[1] - strides[1], 0)
        pad_top = pad_h // 2
        pad_bot = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padding_lax = [(0, 0), (pad_top, pad_bot), (pad_left, pad_right), (0, 0)]
    else:
        padding_lax = [(0, 0), (0, 0), (0, 0), (0, 0)]

    return jax.lax.reduce_window(
        x,
        init_value=-jnp.inf,
        computation=jax.lax.max,
        window_dimensions=(1, window_shape[0], window_shape[1], 1),
        window_strides=(1, strides[0], strides[1], 1),
        padding=padding_lax,
    )


class ResnetStack(nnx.Module):
    """ResNet stack module.

    Applies a convolutional layer followed by optional max-pooling, then
    ``num_blocks`` residual blocks.

    Args:
        in_channels: Number of input channels.
        num_features: Number of output channels for every Conv layer.
        num_blocks: Number of residual blocks to stack.
        max_pooling: Whether to apply 3x3 max-pooling after the first conv.
        rngs: NNX RNG state.

    Example::

        import jax, jax.numpy as jnp
        from flax import nnx
        model = ResnetStack(in_channels=3, num_features=16, num_blocks=2, rngs=nnx.Rngs(0))
        x = jnp.ones((1, 64, 64, 3))
        y = model(x)  # shape: (1, 32, 32, 16)
    """

    def __init__(
        self,
        in_channels: int,
        num_features: int,
        num_blocks: int,
        max_pooling: bool = True,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.max_pooling = max_pooling

        initializer = nnx.initializers.xavier_uniform()

        # Initial conv: maps in_channels → num_features.
        self.init_conv = nnx.Conv(
            in_channels,
            num_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='SAME',
            kernel_init=initializer,
            rngs=rngs,
        )
        self.block_conv1 = [
            nnx.Conv(
                num_features, num_features,
                kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                kernel_init=initializer, rngs=rngs,
            )
            for _ in range(num_blocks)
        ]
        self.block_conv2 = [
            nnx.Conv(
                num_features, num_features,
                kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                kernel_init=initializer, rngs=rngs,
            )
            for _ in range(num_blocks)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        conv_out = self.init_conv(x)

        if self.max_pooling:
            conv_out = _max_pool(conv_out, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        for i in range(self.num_blocks):
            block_input = conv_out
            conv_out = jax.nn.relu(conv_out)
            conv_out = self.block_conv1[i](conv_out)
            conv_out = jax.nn.relu(conv_out)
            conv_out = self.block_conv2[i](conv_out)
            conv_out = conv_out + block_input

        return conv_out


class ImpalaEncoder(nnx.Module):
    """IMPALA encoder.

    Stacks multiple :class:`ResnetStack` blocks then flattens and passes
    through a small MLP, producing a fixed-size embedding from pixel input.

    Args:
        obs_shape: Input observation shape ``(H, W, C)``.
        width: Channel-width multiplier applied to every ``stack_sizes`` entry.
        stack_sizes: Number of features for each ResnetStack stage.
        num_blocks: Residual blocks per stage.
        dropout_rate: Dropout rate; ``None`` disables dropout.
        mlp_hidden_dims: Hidden dimensions for the final MLP.
        layer_norm: Whether to apply layer normalisation after the conv stages
            and inside the MLP.
        rngs: NNX RNG state.

    Example::

        import jax, jax.numpy as jnp
        from flax import nnx
        model = ImpalaEncoder(obs_shape=(64, 64, 3), width=1, rngs=nnx.Rngs(0))
        x = jnp.ones((1, 64, 64, 3), dtype=jnp.uint8)
        y = model(x)  # shape: (1, 512)
    """

    def __init__(
        self,
        obs_shape: tuple,
        width: int = 1,
        stack_sizes: tuple = (16, 32, 32),
        num_blocks: int = 2,
        dropout_rate: float | None = None,
        mlp_hidden_dims: Sequence[int] = (512,),
        layer_norm: bool = False,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.dropout_rate = dropout_rate
        self.layer_norm_flag = layer_norm

        # Track channel count: first stack gets obs channels, subsequent stacks get previous output.
        h, w, in_ch = obs_shape
        stack_list = []
        for i in range(len(stack_sizes)):
            out_ch = stack_sizes[i] * width
            stack_list.append(
                ResnetStack(
                    in_channels=in_ch,
                    num_features=out_ch,
                    num_blocks=num_blocks,
                    rngs=rngs,
                )
            )
            in_ch = out_ch
        self.stack_blocks = stack_list

        if dropout_rate is not None:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

        if layer_norm:
            # Compute spatial output size after all stacks (each max-pool halves spatial dims).
            h, w, _ = obs_shape
            for _ in stack_sizes:
                h = (h + 1) // 2
                w = (w + 1) // 2
            flat_dim = h * w * stack_sizes[-1] * width
            self.post_conv_ln = nnx.LayerNorm(flat_dim, rngs=rngs)
        else:
            self.post_conv_ln = None

        # Compute flat_dim for MLP input regardless of layer_norm.
        h, w, _ = obs_shape
        for _ in stack_sizes:
            h = (h + 1) // 2
            w = (w + 1) // 2
        flat_dim = h * w * stack_sizes[-1] * width

        self.mlp = MLP(flat_dim, mlp_hidden_dims, activate_final=True, layer_norm=layer_norm, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        train: bool = True,
        cond_var: jax.Array | None = None,
    ) -> jax.Array:
        x = x.astype(jnp.float32) / 255.0

        conv_out = x
        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = jax.nn.relu(conv_out)
        if self.post_conv_ln is not None:
            # Flatten before LayerNorm, then pass to MLP.
            out = conv_out.reshape((*x.shape[:-3], -1))
            out = self.post_conv_ln(out)
        else:
            out = conv_out.reshape((*x.shape[:-3], -1))

        out = self.mlp(out)
        return out


class GCEncoder(nnx.Module):
    """Helper module to handle inputs to goal-conditioned networks.

    It takes observations (s) and goals (g) and returns the concatenation of
    ``state_encoder(s)``, ``goal_encoder(g)``, and
    ``concat_encoder([s, g])``.  Encoders that are ``None`` are skipped.

    Args:
        state_encoder: Optional encoder applied to the observation alone.
        goal_encoder: Optional encoder applied to the goal alone.
        concat_encoder: Optional encoder applied to ``[obs, goal]``
            concatenated along the last axis.

    Example::

        import jax, jax.numpy as jnp
        from flax import nnx
        from utils.networks import MLP
        enc = GCEncoder(
            state_encoder=MLP(8, (32,), rngs=nnx.Rngs(0)),
            goal_encoder=MLP(8, (32,), rngs=nnx.Rngs(1)),
        )
        obs, goal = jnp.ones((1, 8)), jnp.ones((1, 8))
        rep = enc(obs, goal)  # shape: (1, 64)
    """

    def __init__(
        self,
        state_encoder=None,
        goal_encoder=None,
        concat_encoder=None,
    ) -> None:
        self.state_encoder = state_encoder
        self.goal_encoder = goal_encoder
        self.concat_encoder = concat_encoder

    def __call__(
        self,
        observations: jax.Array,
        goals: jax.Array | None = None,
        goal_encoded: bool = False,
    ) -> jax.Array:
        """Returns the representations of observations and goals.

        If ``goal_encoded`` is ``True``, ``goals`` is assumed to be already
        encoded representations. In this case, either ``goal_encoder`` or
        ``concat_encoder`` must be ``None``.
        """
        reps = []
        if self.state_encoder is not None:
            reps.append(self.state_encoder(observations))
        if goals is not None:
            if goal_encoded:
                # Can't have both goal_encoder and concat_encoder in this case.
                assert self.goal_encoder is None or self.concat_encoder is None
                reps.append(goals)
            else:
                if self.goal_encoder is not None:
                    reps.append(self.goal_encoder(goals))
                if self.concat_encoder is not None:
                    reps.append(self.concat_encoder(jnp.concatenate([observations, goals], axis=-1)))
        reps = jnp.concatenate(reps, axis=-1)
        return reps


# Factories for encoder_modules: each entry is a callable that takes
# (obs_shape, rngs) and returns an ImpalaEncoder instance.
encoder_modules = {
    'impala': lambda obs_shape, rngs: ImpalaEncoder(obs_shape=obs_shape, width=1, rngs=rngs),
    'impala_debug': lambda obs_shape, rngs: ImpalaEncoder(
        obs_shape=obs_shape, width=1, num_blocks=1, stack_sizes=(4, 4), rngs=rngs
    ),
    'impala_small': lambda obs_shape, rngs: ImpalaEncoder(
        obs_shape=obs_shape, width=1, num_blocks=1, rngs=rngs
    ),
    'impala_large': lambda obs_shape, rngs: ImpalaEncoder(
        obs_shape=obs_shape, width=1, stack_sizes=(64, 128, 128),
        mlp_hidden_dims=(1024,), rngs=rngs
    ),
}
