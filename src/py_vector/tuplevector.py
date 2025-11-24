# ======================================================================
# TupleVector  (Internal backend for PyVector)
# ======================================================================

from datetime import date, datetime
from typing import Any, Iterable, List, Tuple
import math

from .abstract import AbstractVector, VectorOpsMixin
from .typing import DataType, infer_dtype, validate_scalar
from .errors import PyVectorTypeError


# ==========================================================
# Utility helpers
# ==========================================================

def _is_hashable(x: Any) -> bool:
    try:
        hash(x)
        return True
    except Exception:
        return False


def _safe_sortable_list(xs: Iterable[Any]) -> List[Any]:
    """
    Return a deterministically sortable representation (strings if needed).
    Used to make set hashing stable across runs.
    """
    xs = list(xs)
    try:
        return sorted(xs)
    except Exception:
        # Fallback: sort by repr to get deterministic order
        return sorted((repr(x) for x in xs))


# ==========================================================
# MethodProxy  (None-safe)
# ==========================================================

class MethodProxy:
    """
    Elementwise method proxy for things like .upper(), .strip(), .year, etc.
    None stays None.
    """

    def __init__(self, vector: "TupleVector", method_name: str):
        self._vector = vector
        self._method = method_name

    def __call__(self, *args, **kwargs) -> "TupleVector":
        method = self._method
        v = self._vector

        out: List[Any] = []
        for elem in v._storage:
            if elem is None:
                out.append(None)
                continue

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

class TupleVector(VectorOpsMixin, AbstractVector):
    """
    Pure Python backend with tuple storage, fingerprinting, copy-on-write.
    Inherits operations from VectorOpsMixin.
    """

    _FP_P = (1 << 61) - 1
    _FP_B = 1315423911

    def __init__(self, initial: Iterable[Any] = (), dtype: DataType | None = None,
                 name: str | None = None):
        # Normalize dtype
        if dtype is not None and not isinstance(dtype, DataType):
            dtype = DataType(dtype)

        # Empty vectors have None dtype (backwards compatible)
        if dtype is None:
            self._dtype = infer_dtype(initial) if initial else None
        else:
            self._dtype = dtype

        self._name = name
        self._storage: Tuple[Any, ...] = tuple(initial)

        # Lazy fingerprint
        self._fp: int | None = None
        self._fp_powers: List[int] | None = None

    # ======================================================
    # Internal hashing helpers
    # ======================================================

    @staticmethod
    def _hash_element(x: Any) -> int:
        """
        Stable hashing for nested data, sets, None, lists, tuples, etc.
        Uses a rolling hash with fixed P, B.
        """
        P = TupleVector._FP_P
        B = TupleVector._FP_B

        # None gets a fixed constant
        if x is None:
            return 0x9E3779B97F4A7C15

        # Allow nested vectors/objects that expose fingerprint()
        if hasattr(x, "fingerprint"):
            try:
                return int(x.fingerprint())
            except Exception:
                # Fallback to repr if fingerprint misbehaves
                return hash(repr(x))

        # Floats: special NaN handling, otherwise normal hash
        if isinstance(x, float):
            if math.isnan(x):
                return 0xDEADBEEFCAFEBABE
            # Python already normalizes 0.0 and -0.0 hashes
            return hash(x)

        # Sets: order-independent hashing
        if isinstance(x, set):
            rep = _safe_sortable_list(list(x))
            return TupleVector._hash_element(tuple(rep))

        # Lists/tuples: structured rolling hash
        if isinstance(x, (list, tuple)):
            h = 0
            for elem in x:
                h = (h * B + TupleVector._hash_element(elem)) % P
            return h

        # Anything hashable: use its hash
        if _is_hashable(x):
            return hash(x)

        # Fallback: repr-based hash
        return hash(repr(x))

    # ======================================================
    # Fingerprint computation (lazy)
    # ======================================================

    def _compute_fp_powers(self) -> None:
        n = len(self._storage)
        P = self._FP_P
        B = self._FP_B

        if n == 0:
            self._fp_powers = []
            return

        pw = [1] * n
        for i in range(n - 2, -1, -1):
            pw[i] = (pw[i + 1] * B) % P
        self._fp_powers = pw

    def _compute_fingerprint(self) -> int:
        P = self._FP_P
        B = self._FP_B

        total = 0
        for x in self._storage:
            h = self._hash_element(x)
            total = (total * B + h) % P
        return total

    def fingerprint(self) -> int:
        """
        Lazily compute and cache the fingerprint.
        Invalidated on mutation/promotion.
        """
        if self._fp is None:
            if self._fp_powers is None or len(self._fp_powers) != len(self._storage):
                self._compute_fp_powers()
            self._fp = self._compute_fingerprint()
        return self._fp

    # ======================================================
    # AbstractVector protocol
    # ======================================================

    def __len__(self) -> int:
        return len(self._storage)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return self._storage[key]
        if isinstance(key, slice):
            return TupleVector(self._storage[key], dtype=self._dtype, name=self._name)
        raise PyVectorTypeError("Invalid indexing type for TupleVector")

    def schema(self) -> DataType | None:
        return self._dtype

    # ======================================================
    # __getattr__ → MethodProxy
    # ======================================================

    def __getattr__(self, name: str) -> MethodProxy:
        if name.startswith("_"):
            raise AttributeError(name)
        return MethodProxy(self, name)

    # ======================================================
    # Clone and utility methods
    # ======================================================

    def clone_with(self, data: Iterable[Any]) -> "TupleVector":
        return TupleVector(data, dtype=self._dtype, name=self._name)

    def fillna(self, value: Any) -> "TupleVector":
        """
        Replace None entries with `value`, enforcing dtype compatibility.
        """
        if self._dtype is not None:
            try:
                validate_scalar(value, self._dtype)
            except Exception:
                raise PyVectorTypeError(
                    f"fillna() value {value!r} is incompatible with dtype {self._dtype}"
                )

        new_data = [value if x is None else x for x in self._storage]
        return self.clone_with(new_data)

    def cast(self, target_type: type) -> "TupleVector":
        """
        Cast values to target_type.
        Special handling:
            - date: accepts date or ISO 8601 string
            - datetime: accepts datetime or ISO 8601 string
            - None is preserved
        """

        if target_type is date:
            def conv(x: Any) -> Any:
                if x is None:
                    return None
                if isinstance(x, date) and not isinstance(x, datetime):
                    return x
                return date.fromisoformat(str(x))

        elif target_type is datetime:
            def conv(x: Any) -> Any:
                if x is None:
                    return None
                if isinstance(x, datetime):
                    return x
                return datetime.fromisoformat(str(x))

        else:
            def conv(x: Any) -> Any:
                if x is None:
                    return None
                return target_type(x)

        casted = [conv(x) for x in self._storage]
        return self.clone_with(casted)

    def unique(self) -> "TupleVector":
        """
        Return unique values in stable order.

        - Hashable values are tracked with a set for O(N) uniqueness.
        - Unhashable values fall back to equality checks.
        """
        out: List[Any] = []

        hashable_seen: set[Any] = set()
        unhashable_seen: List[Any] = []

        for x in self._storage:
            if _is_hashable(x):
                if x in hashable_seen:
                    continue
                hashable_seen.add(x)
                out.append(x)
            else:
                # Fallback: O(N) equality check
                if any(x == y for y in unhashable_seen):
                    continue
                unhashable_seen.append(x)
                out.append(x)

        return TupleVector(out, dtype=infer_dtype(out) if out else None)

    # ======================================================
    # Copy-on-write mutation
    # ======================================================

    def _set_values(self, updates: list[tuple[int, Any]]) -> "TupleVector":
        """
        updates: list of (index, new_value)
        Copy-on-write update with dtype promotion.
        """
        if not updates:
            return self

        # Validate indices and detect a value that doesn't fit current dtype
        incompatible_val: Any | None = None
        n = len(self._storage)

        for idx, val in updates:
            if not (0 <= idx < n):
                raise PyVectorTypeError(
                    f"Index {idx} out of range for length {n}"
                )
            if self._dtype is not None:
                try:
                    validate_scalar(val, self._dtype)
                except Exception:
                    incompatible_val = val
                    break

        # Promote dtype if needed
        if incompatible_val is not None:
            new_dtype = infer_dtype([incompatible_val])
            self._dtype = new_dtype
            # Recast entire storage to new dtype
            cast_type = new_dtype.kind
            if cast_type is not None:
                recast: List[Any] = []
                for x in self._storage:
                    if x is None:
                        recast.append(None)
                    else:
                        recast.append(cast_type(x))
                self._storage = tuple(recast)
            # Invalidate fingerprint
            self._fp = None
            self._fp_powers = None

        # Apply updates
        data = list(self._storage)
        for idx, val in updates:
            data[idx] = val

        self._storage = tuple(data)
        self._fp = None
        self._fp_powers = None

        return self

    # ======================================================
    # VectorOpsMixin hooks (elementwise and reduce)
    # ======================================================

    def _elementwise(self, func, other):
        """
        Elementwise binary op with configurable null behavior.
        
        If dtype.nullable:
            None behaves like SQL/Pandas null -> result is None.
        Else:
            None is illegal and will raise.
        """
        nullable = (self._dtype is not None and self._dtype.nullable)

        # vector-vector
        if isinstance(other, AbstractVector):
            if len(self) != len(other):
                raise ValueError("Length mismatch in binary operation")
            
            out = []
            for a, b in zip(self._storage, other):
                # Null-propagation OR strict mode
                if nullable:
                    if a is None or b is None:
                        out.append(None)
                    else:
                        out.append(func(a, b))
                else:
                    # strict mode – allow Python's native errors
                    if a is None or b is None:
                        raise TypeError(
                            f"Null value encountered in non-nullable vector {self._name!r}"
                        )
                    out.append(func(a, b))

            return TupleVector(out, dtype=infer_dtype(out), name=None)

        # vector-scalar
        else:
            b = other
            out = []
            for a in self._storage:
                if nullable:
                    if a is None or b is None:
                        out.append(None)
                    else:
                        out.append(func(a, b))
                else:
                    if a is None or b is None:
                        raise TypeError(
                            f"Null value encountered in non-nullable vector {self._name!r}"
                        )
                    out.append(func(a, b))

            return TupleVector(out, dtype=infer_dtype(out), name=None)


    def _elementwise_reverse(self, op, other):
        """
        Perform elementwise op(other, a) with same semantics as _elementwise,
        but argument order is reversed.
        """
        nullable = (self._dtype is not None and self._dtype.nullable)
        out = []

        if isinstance(other, AbstractVector):
            if len(self) != len(other):
                raise ValueError("Length mismatch in reverse binary operation")

            for b, a in zip(other, self._storage):
                if nullable:
                    if a is None or b is None:
                        out.append(None)
                    else:
                        out.append(op(b, a))
                else:
                    if a is None or b is None:
                        raise TypeError(
                            f"Null in non-nullable vector during reverse operation"
                        )
                    out.append(op(b, a))
        else:
            # other is scalar
            b = other
            for a in self._storage:
                if nullable:
                    if a is None or b is None:
                        out.append(None)
                    else:
                        out.append(op(b, a))
                else:
                    if a is None or b is None:
                        raise TypeError(
                            f"Null in non-nullable vector during reverse operation"
                        )
                    out.append(op(b, a))

        return TupleVector(out, dtype=infer_dtype(out), name=None)


    def _reduce(self, func) -> Any:
        """
        Reduction used by VectorOpsMixin (e.g., sum, min, max).
        """
        it = iter(self._storage)
        try:
            acc = next(it)
        except StopIteration:
            raise ValueError("Cannot reduce empty vector")
        for x in it:
            acc = func(acc, x)
        return acc
