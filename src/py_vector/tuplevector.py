# ======================================================================
# TupleVector  (Internal backend for PyVector)
# ======================================================================
"""
Internal storage backend for PyVector implementing:
- Immutable tuple-based storage
- Copy-on-write semantics
- Deterministic fingerprinting for O(1) change detection
- None-safe method proxying
- Robust type promotion
"""

from datetime import date, datetime
from typing import Any, Iterable, List
import math

from .typing import (
    DataType,
    infer_dtype,
    validate_scalar,
)
from .errors import (
    PyVectorTypeError,
    PyVectorIndexError,
    PyVectorValueError
)


# ==========================================================
# Utility helpers
# ==========================================================

def _is_hashable(x):
    """Check if an object is hashable without raising exceptions."""
    try:
        hash(x)
        return True
    except Exception:
        return False


def _safe_sortable_list(xs):
    """Return a deterministically sortable representation (strings if needed)."""
    try:
        return sorted(xs)
    except Exception:
        return sorted((repr(x) for x in xs))


# ==========================================================
# MethodProxy  (Bug #2 fix: None-safe)
# ==========================================================

class MethodProxy:
    """
    Proxy that defers method calls to each element in a TupleVector.
    
    Bug #2 Fix: Handles None elements gracefully by propagating None
    instead of attempting to call methods on None.
    
    Examples
    --------
    >>> v = TupleVector(["hello", None, "world"])
    >>> v.upper()  # Returns TupleVector(["HELLO", None, "WORLD"])
    """
    def __init__(self, vector, method_name):
        self._vector = vector
        self._method = method_name

    def __call__(self, *args, **kwargs):
        method = self._method
        v = self._vector

        out = []
        for elem in v._storage:
            if elem is None:
                # Bug #2 fix: propagate None instead of calling method
                out.append(None)
            else:
                fn = getattr(elem, method, None)
                if fn is None:
                    raise AttributeError(
                        f"Element of type {type(elem).__name__} "
                        f"has no method '{method}'"
                    )
                out.append(fn(*args, **kwargs))

        return v.clone_with(out)


# ==========================================================
# TupleVector Implementation
# ==========================================================

class TupleVector:
    """
    The internal backend for PyVector. Pure Python, immutable storage,
    fingerprinted, copy-on-write.
    
    This class is not intended to be used directly - use PyVector instead.
    
    Key Features:
    - Immutable tuple-based storage
    - Lazy fingerprint computation for O(1) change detection
    - Deterministic hashing (Bug #13)
    - None-safe method proxying (Bug #2)
    - Robust type promotion on mutation (Bug #1)
    """

    # Large Mersenne-like prime for hashing
    _FP_P = (1 << 61) - 1  # Mersenne prime
    _FP_B = 1315423911     # Base for rolling hash

    def __init__(self, initial=(), dtype=None, name=None):
        """
        Initialize TupleVector with data.
        
        Parameters
        ----------
        initial : iterable
            Initial values
        dtype : DataType or type, optional
            Type specification. If None, will be inferred.
        name : str, optional
            Column name
        """
        # ------------------------------------------
        # Schema
        # ------------------------------------------
        if dtype is not None and not isinstance(dtype, DataType):
            dtype = DataType(dtype)
        
        # Infer dtype if not provided and we have data
        if dtype is None and initial:
            dtype = infer_dtype(initial)
        
        self._dtype = dtype
        self._name = name

        # ------------------------------------------
        # Storage
        # ------------------------------------------
        self._storage = tuple(initial)

        # ------------------------------------------
        # Fingerprint cache (Bug #16: Lazy computation)
        # None = invalid/not yet computed
        # ------------------------------------------
        self._fp = None
        self._fp_powers = None  # computed on demand

    # ======================================================
    # Internal hashing helpers (Bug #13: Deterministic sets)
    # ======================================================

    @staticmethod
    def _hash_element(x):
        """
        Stable hashing for nested data, sets, None, lists, tuples, etc.
        
        Bug #13 Fix: Provides deterministic hashing for:
        - Sets (normally order-randomized)
        - NaN floats (normally unhashable)
        - Nested structures
        - None values
        """
        P = TupleVector._FP_P
        B = TupleVector._FP_B

        # None
        if x is None:
            return 0x9E3779B97F4A7C15  # golden ratio constant

        # PyVector/TupleVector instance
        if hasattr(x, "fingerprint"):
            return x.fingerprint()

        # Floats (including NaN)
        if isinstance(x, float):
            if math.isnan(x):
                return 0xDEADBEEFCAFEBABE  # deterministic NaN hash
            return hash(x)

        # Sets (order-randomized normally) - Bug #13 fix
        if isinstance(x, set):
            rep = _safe_sortable_list(list(x))
            return TupleVector._hash_element(tuple(rep))

        # Lists or tuples
        if isinstance(x, (list, tuple)):
            h = 0
            for elem in x:
                h = (h * B + TupleVector._hash_element(elem)) % P
            return h

        # Hashable types
        if _is_hashable(x):
            return hash(x)

        # Fallback: repr
        return hash(repr(x))

    # ======================================================
    # Fingerprint core
    # ======================================================

    def _compute_fp_powers(self):
        """Precompute B^(n-i-1) mod P for incremental fingerprint updates."""
        n = len(self._storage)
        P = self._FP_P
        B = self._FP_B

        pw = [1] * n
        for i in range(n - 2, -1, -1):
            pw[i] = (pw[i + 1] * B) % P
        self._fp_powers = pw

    def _compute_fingerprint(self):
        """Compute full fingerprint from scratch using rolling hash."""
        P = self._FP_P
        B = self._FP_B

        total = 0
        for x in self._storage:
            h = self._hash_element(x)
            total = (total * B + h) % P
        return total

    # ======================================================
    # Public fingerprint API
    # ======================================================

    def fingerprint(self):
        """
        Get cached fingerprint for O(1) change detection.
        
        Bug #16: Lazy computation - only computes when needed.
        Bug #1: Properly invalidated on dtype promotion.
        """
        # Lazy compute
        if self._fp is None:
            # compute powers first if needed
            if self._fp_powers is None or len(self._fp_powers) != len(self._storage):
                self._compute_fp_powers()
            self._fp = self._compute_fingerprint()
        return self._fp

    # ======================================================
    # Basic protocol
    # ======================================================

    def __len__(self):
        return len(self._storage)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._storage[key]
        elif isinstance(key, slice):
            return TupleVector(self._storage[key], dtype=self._dtype, name=self._name)
        else:
            raise PyVectorTypeError("Invalid indexing type")

    def schema(self):
        """Return the DataType schema for this vector."""
        return self._dtype

    # Bug #11: truthiness
    def __bool__(self):
        """
        Prevent ambiguous boolean evaluation.
        
        Bug #11 Fix: Raises ValueError to force explicit .all() or .any().
        """
        raise ValueError(
            "The truth value of a PyVector is ambiguous. "
            "Use .all() or .any()."
        )

    # ======================================================
    # __getattr__ → MethodProxy (Bug #2)
    # ======================================================

    def __getattr__(self, name):
        """
        Enable method chaining on vector elements.
        
        Bug #2 Fix: Returns MethodProxy that handles None gracefully.
        
        Examples
        --------
        >>> v = TupleVector(["hello", "world"])
        >>> v.upper()  # Calls .upper() on each element
        """
        # Avoid recursion for internal attributes
        if name.startswith("_"):
            raise AttributeError(name)
        return MethodProxy(self, name)

    # ======================================================
    # Clone helper
    # ======================================================

    def clone_with(self, data):
        """
        Create a new TupleVector with different data but same metadata.
        
        Preserves dtype and name from original vector.
        """
        return TupleVector(data, dtype=self._dtype, name=self._name)

    # ======================================================
    # fillna (Bug #12)
    # ======================================================

    def fillna(self, value):
        """
        Replace None values with a fill value.
        
        Bug #12 Fix: Validates that fill value is compatible with dtype
        before applying.
        
        Parameters
        ----------
        value : object
            Value to replace Nones with
        
        Returns
        -------
        TupleVector
            New vector with Nones replaced
        
        Raises
        ------
        PyVectorTypeError
            If fill value is incompatible with vector dtype
        """
        # dtype validation
        if self._dtype is not None:
            try:
                validate_scalar(value, self._dtype)
            except Exception:
                raise PyVectorTypeError(
                    f"fillna() value {value!r} is incompatible with dtype {self._dtype}"
                )

        new_data = [value if x is None else x for x in self._storage]
        return self.clone_with(new_data)

    # ======================================================
    # Casting (Bug #7)
    # ======================================================

    def cast(self, target_type):
        """
        Cast vector elements to a target type.
        
        Bug #7 Fix: Handles date/datetime ISO string conversion properly.
        
        Parameters
        ----------
        target_type : type
            Target Python type (int, float, str, date, datetime, etc.)
        
        Returns
        -------
        TupleVector
            New vector with cast values
        """
        if target_type is date:
            def conv(x):
                if x is None:
                    return None
                if isinstance(x, date):
                    return x
                return date.fromisoformat(x)
        elif target_type is datetime:
            def conv(x):
                if x is None:
                    return None
                if isinstance(x, datetime):
                    return x
                return datetime.fromisoformat(x)
        else:
            def conv(x):
                if x is None:
                    return None
                return target_type(x)

        return self.clone_with([conv(x) for x in self._storage])

    # ======================================================
    # unique() (Bug #14)
    # ======================================================

    def unique(self):
        """
        Return unique values in order of first appearance.
        
        Bug #14 Fix: Handles both hashable and unhashable elements correctly
        using fallback equality check for unhashable types.
        
        Returns
        -------
        TupleVector
            New vector containing only unique values
        """
        seen = []
        out = []
        for x in self._storage:
            if _is_hashable(x):
                # Fast path: use hash-based lookup
                if x not in seen:
                    seen.append(x)
                    out.append(x)
            else:
                # Slow O(N^2) fallback for unhashable elements
                if not any((x == y) for y in seen):
                    seen.append(x)
                    out.append(x)
        return TupleVector(out, dtype=infer_dtype(out) if out else None)

    # ======================================================
    # Mutation (copy-on-write) — with Bug #1 fix
    # ======================================================

    def _set_values(self, updates):
        """
        Apply copy-on-write updates to vector.
        
        Bug #1 Fix: Properly invalidates fingerprint and powers when dtype
        promotion occurs. Previously, fingerprint was only invalidated after
        updates, missing the promotion case.
        
        Parameters
        ----------
        updates : list of (index, new_value) tuples
            Updates to apply
        
        Returns
        -------
        self
            Returns self for chaining (mutations are in-place on this instance)
        """
        if not updates:
            return self

        # 1. If promotion needed, detect BEFORE applying
        # This ensures we promote the dtype first, then apply updates
        incompatible_val = None
        if self._dtype is not None:
            for idx, val in updates:
                try:
                    validate_scalar(val, self._dtype)
                except Exception:
                    incompatible_val = val
                    break

        if incompatible_val is not None:
            # Promote dtype to accommodate new value
            promoted = self._dtype.promote_with(incompatible_val)
            self._dtype = promoted
            
            # BUG #1 fix: invalidate fp and powers DURING promotion
            self._fp = None
            self._fp_powers = None

        # 2. Now apply updates
        data = list(self._storage)
        for idx, val in updates:
            data[idx] = val

        self._storage = tuple(data)

        # 3. Invalidate fingerprint after updates
        self._fp = None
        self._fp_powers = None

        return self

    # ======================================================
    # Elementwise ops (will be used by VectorOpsMixin later)
    # ======================================================

    def _elementwise(self, func, other):
        """
        Apply binary function element-wise.
        
        Used internally for arithmetic operations. Will be called by
        VectorOpsMixin in future refactoring.
        
        Parameters
        ----------
        func : callable
            Binary function to apply
        other : TupleVector or scalar
            Right operand
        
        Returns
        -------
        TupleVector
            New vector with results
        """
        if isinstance(other, TupleVector):
            out = [func(a, b) for a, b in zip(self._storage, other._storage)]
        else:
            out = [func(a, other) for a in self._storage]
        return TupleVector(out, dtype=infer_dtype(out) if out else None, name=self._name)

    def _reduce(self, func):
        """
        Reduce vector using binary function.
        
        Parameters
        ----------
        func : callable
            Binary reduction function
        
        Returns
        -------
        object
            Reduced scalar value
        
        Raises
        ------
        ValueError
            If vector is empty
        """
        it = iter(self._storage)
        try:
            acc = next(it)
        except StopIteration:
            raise ValueError("Cannot reduce empty vector")
        for x in it:
            acc = func(acc, x)
        return acc
