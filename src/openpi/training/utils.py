from collections.abc import Callable
from typing import Any

from flax import nnx
from flax import struct
import jax
import jax.numpy as jnp
import optax

from openpi.models import model as _model
from openpi.shared import array_typing as at


def ema_update(ema_params: at.Params, new_params: at.Params, decay: float) -> at.Params:
    """EMA over the float leaves of a param tree, carrying every other leaf through.

    `nnx.state(model)` is not purely float weights: a module holding an `nnx.Rngs` (the
    event head's `nnx.Dropout` does) contributes a `key<fry>` PRNG key and a `uint32`
    counter to the same tree. Multiplying those by `decay` raises
    `TypeError: multiply does not accept dtypes float32, key<fry>`, so a plain
    `jax.tree.map` over the whole tree only works for models whose state happens to be
    all-float. Averaging RNG state would be meaningless anyway — the shadow copy takes
    the live value.

    Only reachable with `event_tracking=True` and `ema_decay` set, which is why upstream
    (LoRA everywhere, `ema_decay=None`) never hit it.
    """

    def _update(old, new):
        if not jnp.issubdtype(new.dtype, jnp.floating):
            return new
        return decay * old + (1 - decay) * new

    return jax.tree.map(_update, ema_params, new_params)


@at.typecheck
@struct.dataclass
class TrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")
