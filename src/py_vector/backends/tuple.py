# src/py_vector/backends/tuple.py
"""
TupleBackend
============

Pure-Python, zero-dependency backend for PyVector.

- Stores data as an immutable tuple (self._data)
- Handles dtype + nullability via DataType / infer_dtype / validate_scalar
- Provides arithmetic, comparison, casting, fillna, unique, fingerprint
- Supports copy-on-write updates with dtype promotion via set_values()
- Designed so PyVector (front-end) pays ONE extra function call per op,
  not per element.

This backend is the "fast default" and can be swapped out via config.set_backend().
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable, List, Tuple
import math

# Adjust these imports to match your exact file structure
from ..typing import DataType, infer_dtype, validate_scalar
from ..errors import PyVectorTypeError, PyVectorValueError, PyVectorIndexError
from ..config import register_backend


# ============================================================
# Small helpers
# ============================================================

def _is_hashable(x: Any) -> bool:
    try:
        hash(x)
        return True
    except Exception:
        return False


def _safe_sortable_list(xs: Iterable[Any]) -> List[Any]:
    """
    Deterministic representation for sets in fingerprinting.
    """
    try:
        return sorted(xs)
    except Exception:
        return sorted((repr(x) for x in xs))


# ============================================================
# Main backend
# ============================================================

class TupleBackend:
    """
    Pure-Python, tuple-backed implementation of a vector.
    """

    # Optimization: Prevent __dict__ creation to save RAM
    # Must include ALL instance variables used in __init__
    __slots__ = ('_data', '_dtype', '_name', '_fp', '_fp_powers')

    _FP_P = (1 << 61) - 1
    _FP_B = 1315423911

    # --------------------------------------------------------
    # Construction
    # --------------------------------------------------------

    def __init__(
        self,
        data: Iterable[Any] | None = None,
        dtype: DataType | type | None = None,
        name: str | None = None,
    ) -> None:
        if data is None:
            data = ()

        # Store data as-is (including PyVector wrappers for PyTable columns)
        self._data = tuple(data)

        if isinstance(dtype, DataType):
            self._dtype: DataType | None = dtype
        elif dtype is None:
            self._dtype = infer_dtype(self._data) if self._data else None
        else:
            self._dtype = DataType(dtype)

        self._name: str | None = name

        # Fingerprint cache + powers
        self._fp: int | None = None
        self._fp_powers: List[int] | None = None

    # --------------------------------------------------------
    # Basic protocol
    # --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._data[key]
        
        if isinstance(key, slice):
            return TupleBackend(self._data[key], dtype=self._dtype, name=None)

        # Handle Lists/Tuples (Boolean Mask or Integer Gather)
        if isinstance(key, (list, tuple)):
            if not key:
                 return TupleBackend([], dtype=self._dtype, name=None)
            
            # Case A: Boolean Mask
            if isinstance(key[0], bool):
                 if len(key) != len(self._data):
                     raise PyVectorValueError(f"Mask length {len(key)} != vector length {len(self._data)}")
                 new_data = [x for x, m in zip(self._data, key) if m]
                 return TupleBackend(new_data, dtype=self._dtype, name=None)
            
            # Case B: Integer Gather
            if isinstance(key[0], int):
                 # Handle negative indices for list access
                 n = len(self._data)
                 new_data = [self._data[i + n if i < 0 else i] for i in key]
                 return TupleBackend(new_data, dtype=self._dtype, name=None)

        raise PyVectorTypeError(f"Invalid index type {type(key).__name__} for TupleBackend")

    # Accessors
    @property
    def data(self) -> Tuple[Any, ...]:
        return self._data

    @property
    def dtype(self) -> DataType | None:
        return self._dtype

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, value: str | None):
        self._name = value

    def to_list(self) -> List[Any]:
        return list(self._data)

    # --------------------------------------------------------
    # Fingerprinting
    # --------------------------------------------------------

    @staticmethod
    def _hash_element(x: Any) -> int:
        P = TupleBackend._FP_P
        B = TupleBackend._FP_B

        if x is None:
            return 0x9E3779B97F4A7C15
        
        if hasattr(x, "fingerprint") and callable(getattr(x, "fingerprint")):
            return int(x.fingerprint())

        if isinstance(x, float):
            if math.isnan(x):
                return 0xDEADBEEFCAFEBABE
            return hash(x)

        if isinstance(x, set):
            rep = _safe_sortable_list(list(x))
            return TupleBackend._hash_element(tuple(rep))

        if isinstance(x, (list, tuple)):
            h = 0
            for elem in x:
                h = (h * B + TupleBackend._hash_element(elem)) % P
            return h

        if _is_hashable(x):
            return hash(x)

        return hash(repr(x))

    def _ensure_fp_powers(self) -> None:
        n = len(self._data)
        if n == 0:
            self._fp_powers = []
            return
        P = self._FP_P
        B = self._FP_B
        pw = [1] * n
        for i in range(n - 2, -1, -1):
            pw[i] = (pw[i + 1] * B) % P
        self._fp_powers = pw

    def _compute_fingerprint_full(self) -> int:
        P = self._FP_P
        B = self._FP_B
        total = 0
        for x in self._data:
            h = self._hash_element(x)
            total = (total * B + h) % P
        return total

    def fingerprint(self) -> int:
        if self._fp is None:
            if self._fp_powers is None or len(self._fp_powers) != len(self._data):
                self._ensure_fp_powers()
            self._fp = self._compute_fingerprint_full()
        return self._fp

    def _invalidate_fp(self) -> None:
        self._fp = None

    # --------------------------------------------------------
    # DataType & Mutation
    # --------------------------------------------------------

    def _promote(self, new_dtype: DataType) -> None:
        # [FIX] Handle raw types passed by tests (e.g. float vs DataType(float))
        if not isinstance(new_dtype, DataType):
            new_dtype = DataType(new_dtype)

        self._dtype = new_dtype
        if new_dtype is None:
            return

        cast = getattr(new_dtype, "cast", None)
        if cast is None:
            pytype = new_dtype.kind
            def cast(x):
                return x if isinstance(x, pytype) else pytype(x)

        new_data = []
        for x in self._data:
            if x is None:
                if getattr(new_dtype, "nullable", False):
                    new_data.append(None)
                else:
                    raise PyVectorTypeError(
                        f"Cannot promote to non-nullable {new_dtype} with None values."
                    )
            else:
                new_data.append(cast(x))

        self._data = tuple(new_data)
        self._invalidate_fp()

    def copy(self) -> "TupleBackend":
        return TupleBackend(
            data=self._data,
            dtype=self._dtype,
            name=self._name
        )


    def set_values(self, updates: list[tuple[int, Any]]) -> "TupleBackend":
        if not updates:
            return self

        incompatible_value = None
        if self._dtype is not None:
            for idx, val in updates:
                try:
                    validate_scalar(val, self._dtype)
                except Exception:
                    incompatible_value = val
                    break

        if incompatible_value is not None:
            new_dtype = infer_dtype(list(self._data) + [incompatible_value])
            try:
                self._promote(new_dtype)
            except Exception as e:
                raise PyVectorTypeError(f"Promotion failed: {e}") from e

        data = list(self._data)
        n = len(data)
        for idx, val in updates:
            if not (0 <= idx < n):
                raise PyVectorIndexError(f"Index {idx} out of range")
            data[idx] = val

        self._data = tuple(data)
        self._invalidate_fp()
        return self

    # --------------------------------------------------------
    # Arithmetic (Unrolled for Speed)
    # --------------------------------------------------------
    # These return TupleBackend. PyVector wraps them.

    def add(self, other) -> "TupleBackend":
        if isinstance(other, TupleBackend):
            if len(self._data) != len(other._data):
                raise PyVectorValueError(f"Length mismatch: {len(self._data)} vs {len(other._data)}")
            # FAST PATH: No lambda, explicit logic
            out = [
                (a + b) if (a is not None and b is not None) else None 
                for a, b in zip(self._data, other._data)
            ]
        else:
            # Scalar path
            out = [
                (a + other) if (a is not None and other is not None) else None 
                for a in self._data
            ]
        return TupleBackend(out, dtype=infer_dtype(out), name=None)

    def sub(self, other) -> "TupleBackend":
        if isinstance(other, TupleBackend):
            if len(self._data) != len(other._data):
                raise PyVectorValueError(f"Length mismatch")
            out = [
                (a - b) if (a is not None and b is not None) else None 
                for a, b in zip(self._data, other._data)
            ]
        else:
            out = [
                (a - other) if (a is not None and other is not None) else None 
                for a in self._data
            ]
        return TupleBackend(out, dtype=infer_dtype(out), name=None)

    def mul(self, other) -> "TupleBackend":
        if isinstance(other, TupleBackend):
            if len(self._data) != len(other._data):
                raise PyVectorValueError(f"Length mismatch")
            out = [
                (a * b) if (a is not None and b is not None) else None 
                for a, b in zip(self._data, other._data)
            ]
        else:
            out = [
                (a * other) if (a is not None and other is not None) else None 
                for a in self._data
            ]
        return TupleBackend(out, dtype=infer_dtype(out), name=None)

    def truediv(self, other) -> "TupleBackend":
        if isinstance(other, TupleBackend):
            if len(self._data) != len(other._data):
                raise PyVectorValueError(f"Length mismatch")
            out = [
                (a / b) if (a is not None and b is not None) else None 
                for a, b in zip(self._data, other._data)
            ]
        else:
            out = [
                (a / other) if (a is not None and other is not None) else None 
                for a in self._data
            ]
        return TupleBackend(out, dtype=infer_dtype(out), name=None)

    def floordiv(self, other) -> "TupleBackend":
        if isinstance(other, TupleBackend):
            if len(self._data) != len(other._data):
                raise PyVectorValueError(f"Length mismatch")
            out = [
                (a // b) if (a is not None and b is not None) else None 
                for a, b in zip(self._data, other._data)
            ]
        else:
            out = [
                (a // other) if (a is not None and other is not None) else None 
                for a in self._data
            ]
        return TupleBackend(out, dtype=infer_dtype(out), name=None)

    def mod(self, other) -> "TupleBackend":
        if isinstance(other, TupleBackend):
            if len(self._data) != len(other._data):
                raise PyVectorValueError(f"Length mismatch")
            out = [
                (a % b) if (a is not None and b is not None) else None 
                for a, b in zip(self._data, other._data)
            ]
        else:
            out = [
                (a % other) if (a is not None and other is not None) else None 
                for a in self._data
            ]
        return TupleBackend(out, dtype=infer_dtype(out), name=None)

    def pow(self, other) -> "TupleBackend":
        if isinstance(other, TupleBackend):
            if len(self._data) != len(other._data):
                raise PyVectorValueError(f"Length mismatch")
            out = [
                (a ** b) if (a is not None and b is not None) else None 
                for a, b in zip(self._data, other._data)
            ]
        else:
            out = [
                (a ** other) if (a is not None and other is not None) else None 
                for a in self._data
            ]
        return TupleBackend(out, dtype=infer_dtype(out), name=None)

    # --------------------------------------------------------
    # Reverse Arithmetic (Scalar only usually)
    # --------------------------------------------------------
    
    def radd(self, other) -> "TupleBackend":
        # a + b == b + a
        return self.add(other)

    def rsub(self, other) -> "TupleBackend":
        out = [
            (other - a) if (a is not None and other is not None) else None 
            for a in self._data
        ]
        return TupleBackend(out, dtype=infer_dtype(out), name=None)

    def rmul(self, other) -> "TupleBackend":
        return self.mul(other)

    def rtruediv(self, other) -> "TupleBackend":
        out = [
            (other / a) if (a is not None and other is not None) else None 
            for a in self._data
        ]
        return TupleBackend(out, dtype=infer_dtype(out), name=None)

    def rfloordiv(self, other) -> "TupleBackend":
        out = [
            (other // a) if (a is not None and other is not None) else None 
            for a in self._data
        ]
        return TupleBackend(out, dtype=infer_dtype(out), name=None)

    def rmod(self, other) -> "TupleBackend":
        out = [
            (other % a) if (a is not None and other is not None) else None 
            for a in self._data
        ]
        return TupleBackend(out, dtype=infer_dtype(out), name=None)

    def rpow(self, other) -> "TupleBackend":
        out = [
            (other ** a) if (a is not None and other is not None) else None 
            for a in self._data
        ]
        return TupleBackend(out, dtype=infer_dtype(out), name=None)

    # --------------------------------------------------------
    # Comparisons
    # --------------------------------------------------------
    
    def eq(self, other) -> list[bool]:
        if isinstance(other, TupleBackend):
            if len(self._data) != len(other._data):
                raise PyVectorValueError(f"Length mismatch")
            return [(a == b) for a, b in zip(self._data, other._data)]
        return [(a == other) for a in self._data]

    def ne(self, other) -> list[bool]:
        if isinstance(other, TupleBackend):
            if len(self._data) != len(other._data):
                raise PyVectorValueError(f"Length mismatch")
            return [(a != b) for a, b in zip(self._data, other._data)]
        return [(a != other) for a in self._data]

    def lt(self, other) -> list[bool]:
        if isinstance(other, TupleBackend):
            if len(self._data) != len(other._data):
                raise PyVectorValueError(f"Length mismatch")
            return [(a < b) for a, b in zip(self._data, other._data)]
        return [(a < other) for a in self._data]

    def le(self, other) -> list[bool]:
        if isinstance(other, TupleBackend):
            if len(self._data) != len(other._data):
                raise PyVectorValueError(f"Length mismatch")
            return [(a <= b) for a, b in zip(self._data, other._data)]
        return [(a <= other) for a in self._data]

    def gt(self, other) -> list[bool]:
        if isinstance(other, TupleBackend):
            if len(self._data) != len(other._data):
                raise PyVectorValueError(f"Length mismatch")
            return [(a > b) for a, b in zip(self._data, other._data)]
        return [(a > other) for a in self._data]

    def ge(self, other) -> list[bool]:
        if isinstance(other, TupleBackend):
            if len(self._data) != len(other._data):
                raise PyVectorValueError(f"Length mismatch")
            return [(a >= b) for a, b in zip(self._data, other._data)]
        return [(a >= other) for a in self._data]

    # --------------------------------------------------------
    # Reductions
    # --------------------------------------------------------

    def sum(self) -> Any:
        values = [v for v in self._data if v is not None]
        if not values:
            return 0
        return sum(values)

    def min(self) -> Any:
        values = [v for v in self._data if v is not None]
        if not values:
            return None
        return min(values)

    def max(self) -> Any:
        values = [v for v in self._data if v is not None]
        if not values:
            return None
        return max(values)

    def mean(self) -> Any:
        values = [v for v in self._data if v is not None]
        if not values:
            return None
        return sum(values) / len(values)

    def stdev(self) -> float | None:
        # Filter Nones
        values = [v for v in self._data if v is not None]
        n = len(values)
        if n < 2:
            return None
        
        # Standard Variance Formula
        avg = sum(values) / n
        variance = sum((x - avg) ** 2 for x in values) / (n - 1)
        return math.sqrt(variance)

    def all(self) -> bool:
        return all(self._data)

    def any(self) -> bool:
        return any(self._data)

    # --------------------------------------------------------
    # NA / casting / unique
    # --------------------------------------------------------

    def isna(self) -> list[bool]:
        return [x is None for x in self._data]

    def dropna(self) -> "TupleBackend":
        filtered = [x for x in self._data if x is not None]
        return TupleBackend(filtered, dtype=infer_dtype(filtered), name=None)

    def fillna(self, value: Any) -> "TupleBackend":
        if self._dtype is not None:
            try:
                validate_scalar(value, self._dtype)
            except Exception as e:
                raise PyVectorTypeError(f"Incompatible fillna value: {e}") from e

        new_data = [value if x is None else x for x in self._data]
        new_dtype = self._dtype.with_nullable(False) if self._dtype else infer_dtype(new_data)
        
        return TupleBackend(new_data, dtype=new_dtype, name=None)

    def cast(self, target_type: type | DataType) -> "TupleBackend":
        if isinstance(target_type, DataType):
            target_dtype = target_type
            target_pytype = target_type.kind
        else:
            target_dtype = DataType(target_type)
            target_pytype = target_type

        def conv(x):
            if x is None:
                return None
            if target_pytype is date:
                if isinstance(x, date) and not isinstance(x, datetime):
                    return x
                return date.fromisoformat(str(x))
            if target_pytype is datetime:
                if isinstance(x, datetime):
                    return x
                return datetime.fromisoformat(str(x))
            return target_pytype(x)

        new_data = [conv(x) for x in self._data]
        return TupleBackend(new_data, dtype=target_dtype, name=None)

    def unique(self) -> "TupleBackend":
        seen_hashable = set()
        seen_unhashable: list[Any] = []
        out: list[Any] = []

        for x in self._data:
            if _is_hashable(x):
                if x not in seen_hashable:
                    seen_hashable.add(x)
                    out.append(x)
            else:
                if not any(x == y for y in seen_unhashable):
                    seen_unhashable.append(x)
                    out.append(x)

        return set(out)


# ============================================================
# Register this as the default backend
# ============================================================

register_backend("python", TupleBackend, priority=0)