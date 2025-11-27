from .registry import BackendRegistry
from .tuple_backend import TupleBackend

def dispatch_map_binary(left, right, func):
    L = type(left)
    R = type(right)

    # Case A: identical → use fast path if available
    if L is R:
        if hasattr(left, "fast_map_binary"):
            return left.fast_map_binary(right, func)

    # Case B: both registered — negotiate "best" backend
    best_type = BackendRegistry.best_backend_for([L, R])
    if best_type is L and hasattr(left, "fast_map_binary"):
        return left.fast_map_binary(right, func)
    if best_type is R and hasattr(right, "fast_map_binary"):
        return right.fast_map_binary(left, func)

    # Case C: fallback → slow python list path
    l = left.to_pylist()
    r = right.to_pylist()
    return TupleBackend([func(a, b) for a, b in zip(l, r)])
