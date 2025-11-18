# py_vector/_alias_tracker.py

import weakref


class AliasError(Exception):
    pass


class _AliasTracker:
    """
    Tracks which PyVector instances reference the same underlying tuple.

    Key design points:
    - Uses id(tuple) → list[weakref.ref(PyVector)]
    - No hashing of PyVector objects (they remain unhashable)
    - No use of WeakSet (because that requires hashing)
    - Pure identity tracking
    - Auto-prunes dead references
    """

    def __init__(self):
        # tuple_id -> list of weakrefs to PyVectors using that tuple
        self._registry = {}

    def _cleanup_dead_refs(self, refs):
        """Remove any weakrefs that no longer point to a live object."""
        return [r for r in refs if r() is not None]

    def register(self, vec, tuple_id):
        """
        Register a PyVector as sharing the tuple with key tuple_id.
        """
        refs = self._registry.setdefault(tuple_id, [])
        
        # clean dead refs before adding
        refs = self._cleanup_dead_refs(refs)

        # Check if vec is already registered for this tuple_id
        for r in refs:
            if r() is vec:
                # Already registered, don't add again
                return

        # store the cleaned list back
        self._registry[tuple_id] = refs

        # we always register by weakref, no hashing required
        refs.append(weakref.ref(vec))

    def unregister(self, vec, tuple_id):
        """
        Remove a PyVector from the registry for a given tuple_id.
        If that tuple_id has no remaining owners, delete the entry.
        """
        refs = self._registry.get(tuple_id)
        if not refs:
            return

        # Remove matching weakref
        alive = []
        for r in refs:
            obj = r()
            if obj is None:
                continue
            if obj is vec:
                continue
            alive.append(r)

        if alive:
            self._registry[tuple_id] = alive
        else:
            del self._registry[tuple_id]

    def check_writable(self, vec, tuple_id):
        """
        Returns True if vec is the *only* owner of tuple_id.
        Otherwise raises AliasError.
        """
        refs = self._registry.get(tuple_id)
        if not refs:
            return True  # nothing registered → writable

        # drop dead weakrefs
        alive = self._cleanup_dead_refs(refs)
        self._registry[tuple_id] = alive

        # Count how many PyVectors still alive share this tuple
        owners = [r() for r in alive if r() is not None]

        if len(owners) <= 1:
            return True

        # >1 owner → alias detected
        raise AliasError(
            "This PyVector shares underlying storage with another.\n"
            "Use .copy() to create an independent, writable vector."
        )


# THE SINGLETON (imported by PyVector)
_ALIAS_TRACKER = _AliasTracker()
