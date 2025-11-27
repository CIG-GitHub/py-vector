class BackendRegistry:
    """
    Global registry of vector backends.
    Allows negotiation (priority), type discovery,
    and strict separation between backend implementations.
    """
    _vector_backends = []

    @classmethod
    def register(cls, backend_type, priority=0):
        """
        backend_type: Backend subclass
        priority: higher = preferred when both operands need promotion
        """
        cls._vector_backends.append((priority, backend_type))
        cls._vector_backends.sort(reverse=True)  # highest priority first

    @classmethod
    def best_backend_for(cls, candidates):
        """
        Given a list of backend *types*, return the best one.

        candidates = [type(left), type(right)]
        """
        for priority, backend_type in cls._vector_backends:
            if backend_type in candidates:
                return backend_type
        raise RuntimeError("No backend registered for candidates.")
