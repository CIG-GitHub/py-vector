# ======================================================================
# Backend Registry + Priority Management
# ======================================================================

_BACKENDS = {}             # name → backend class
_ACTIVE_BACKEND = None     # the backend class currently in use
_ACTIVE_PRIORITY = -10**9  # lowest possible priority
_FORCE_BACKEND = None      # locked backend chosen by user


def register_backend(name: str, backend_cls, priority: int = 0):
    """
    Called by each backend module when imported.

    - Registers the backend.
    - If user has NOT manually forced a backend:
         The highest-priority backend becomes active.
    """
    global _ACTIVE_BACKEND, _ACTIVE_PRIORITY

    _BACKENDS[name] = backend_cls

    # User has locked a backend → ignore priority
    if _FORCE_BACKEND is not None:
        return

    # Otherwise: king-of-the-hill priority selection
    if priority > _ACTIVE_PRIORITY:
        _ACTIVE_BACKEND = backend_cls
        _ACTIVE_PRIORITY = priority


def set_backend(name: str):
    """
    User override: permanently locks the backend.

    Example:
        set_backend("tuple")
    """
    global _FORCE_BACKEND, _ACTIVE_BACKEND

    if name not in _BACKENDS:
        raise KeyError(f"Backend '{name}' is not registered.")

    cls = _BACKENDS[name]
    _FORCE_BACKEND = cls
    _ACTIVE_BACKEND = cls


def get_backend():
    """Return the currently active backend class."""
    if _ACTIVE_BACKEND is None:
        raise RuntimeError(
            "No backend registered. "
            "Ensure the default TupleBackend was imported."
        )
    return _ACTIVE_BACKEND


def available_backends():
    """Return dict of registered backends."""
    return dict(_BACKENDS)
