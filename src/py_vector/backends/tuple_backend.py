from .base_backend import Backend

class TupleBackend(Backend):
    """Your current PyVector._underlying behavior."""

    def __init__(self, data):
        # ALWAYS store tuples internally
        self.data = tuple(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return TupleBackend(self.data[idx])
        if isinstance(idx, list):
            return TupleBackend([self.data[i] for i in idx])
        return self.data[idx]

    def to_pylist(self):
        return list(self.data)

    #
    # Fast-path binary operations (tuple vs tuple)
    #
    def fast_map_binary(self, other, func):
        return TupleBackend((func(a, b) for a, b in zip(self.data, other.data)))


# register with lowest priority
BackendRegistry.register(TupleBackend, priority=0)