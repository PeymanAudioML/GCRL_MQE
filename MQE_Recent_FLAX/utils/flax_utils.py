import dataclasses
import functools
import glob
import os
from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx


class ModuleDict:
    """A plain-Python dictionary of NNX modules.

    This is a thin wrapper that holds a dict of modules and provides a
    convenient interface for TrainState. It is not itself an nnx.Module.

    Attributes:
        modules: Dictionary of modules.
    """

    def __init__(self, modules: Dict[str, Any]) -> None:
        self._modules = modules

    def items(self):
        return self._modules.items()

    def __getitem__(self, key):
        return self._modules[key]


class Sequential(nnx.Module):
    """Sequential container that applies layers in order.

    Args:
        layers: List of callables / nnx.Modules to apply in sequence.

    Example::

        import jax.numpy as jnp
        from flax import nnx
        seq = Sequential([nnx.Linear(4, 8, rngs=nnx.Rngs(0)), jax.nn.relu])
        y = seq(jnp.ones((2, 4)))  # shape: (2, 8)
    """

    def __init__(self, layers: list) -> None:
        self.layers = nnx.List(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=['step', 'params', 'opt_state'],
    meta_fields=['graphdefs', 'tx'],
)
@dataclasses.dataclass(frozen=True)
class TrainState:
    """Custom train state for NNX models.

    Attributes:
        step: Counter tracking training steps; incremented after each
            ``apply_gradients`` call.
        graphdefs: Static dict mapping ``'modules_<name>'`` to
            ``nnx.GraphDef`` — not traced by JAX.
        params: Dynamic dict mapping ``'modules_<name>'`` to
            ``nnx.State`` — traced by JAX.
        tx: optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    graphdefs: Any  # dict[str, nnx.GraphDef] — STATIC
    params: Any     # dict[str, nnx.State]    — DYNAMIC
    tx: Any = None
    opt_state: Any = None

    def replace(self, **kwargs) -> 'TrainState':
        return dataclasses.replace(self, **kwargs)

    @classmethod
    def create(cls, modules, tx=None, **kwargs) -> 'TrainState':
        """Create a new TrainState.

        Args:
            modules: Either a ``ModuleDict`` or a plain ``dict`` mapping
                string names to ``nnx.Module`` instances.
            tx: Optional optax optimizer.
            **kwargs: Additional fields forwarded to the dataclass.
        """
        # Accept both ModuleDict and plain dict.
        if isinstance(modules, ModuleDict):
            module_items = modules.items()
        else:
            module_items = modules.items()

        graphdefs: dict = {}
        states: dict = {}
        for name, module in module_items:
            gd, state = nnx.split(module, nnx.Param)
            key = f'modules_{name}'
            graphdefs[key] = gd
            states[key] = state

        opt_state = tx.init(states) if tx is not None else None
        return cls(
            step=1,
            graphdefs=graphdefs,
            params=states,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, name: str | None = None, params=None, **kwargs):
        """Forward pass through a named module.

        Args:
            *args: Positional arguments forwarded to the module.
            name: Module name (key without the ``'modules_'`` prefix).
            params: Optional parameter dict (``dict[str, nnx.State]``) to use
                instead of the stored parameters. Pass the traced grad_params
                here to flow gradients.
            **kwargs: Keyword arguments forwarded to the module.
        """
        p = params if params is not None else self.params
        key = f'modules_{name}'
        model = nnx.merge(self.graphdefs[key], p[key])
        return model(*args, **kwargs)

    def select(self, name: str):
        """Return a partial that calls the named module."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs) -> 'TrainState':
        """Apply gradients and return an updated TrainState."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        """Compute gradients from ``loss_fn`` and apply them.

        It additionally computes gradient statistics and adds them to the
        returned info dictionary.

        Args:
            loss_fn: Callable that takes ``params`` and returns
                ``(loss, info_dict)``.
        """
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

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

        final_grad_max = jnp.max(grad_max_flat)
        final_grad_min = jnp.min(grad_min_flat)
        final_grad_norm = jnp.linalg.norm(grad_norm_flat, ord=1)

        info.update(
            {
                'grad/max': final_grad_max,
                'grad/min': final_grad_min,
                'grad/norm': final_grad_norm,
            }
        )

        return self.apply_gradients(grads=grads), info


def save_agent(agent, save_dir, epoch):
    """Save the agent to a file.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """
    save_path = os.path.join(save_dir, f'params_{epoch}')
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(save_path, args=ocp.args.StandardSave(agent))
    print(f'Saved to {save_path}')


def restore_agent(agent, restore_path, restore_epoch):
    """Restore the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    path = candidates[0] + f'/params_{restore_epoch}'
    checkpointer = ocp.StandardCheckpointer()
    agent = checkpointer.restore(path, args=ocp.args.StandardRestore(agent))

    print(f'Restored from {path}')

    return agent
