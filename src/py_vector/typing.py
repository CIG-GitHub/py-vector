"""
DataType system for PyVector / PyTable.

Pure metadata design:
  - DataType describes column semantics (type + nullable flag)
  - Null masks live on PyVector instances, not in DataType
  - Promotion is functional (immutable DataType instances)
  - Backend-agnostic and stable
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Optional, Type
import warnings


@dataclass(frozen=True)
class DataType:
    """
    Describes the semantic type of a PyVector column.

    Attributes
    ----------
    kind : Type
        Python type (int, float, str, date, etc.)
    nullable : bool
        Whether the column may contain None values

    Notes
    -----
    - DataType holds zero instance data (no masks, no defaults)
    - Promotion never mutates — always returns new DataType
    - This is backend-agnostic and forms the semantic core
    
    Examples
    --------
    >>> DataType(int)
    DataType(kind=<class 'int'>, nullable=False)
    >>> DataType(int, nullable=True)
    DataType(kind=<class 'int'>, nullable=True)
    >>> DataType(float).promote_with(None)
    DataType(kind=<class 'float'>, nullable=True)
    """

    kind: Type[Any]
    nullable: bool = False

    def __repr__(self):
        if self.nullable:
            return f"<{self.kind.__name__} nullable>"
        return f"<{self.kind.__name__}>"

    @property
    def is_numeric(self) -> bool:
        """True if kind is bool, int, float, or complex."""
        try:
            return issubclass(self.kind, (int, float, complex, bool))
        except TypeError:
            return False

    @property
    def is_temporal(self) -> bool:
        """True if kind is date or datetime."""
        try:
            return issubclass(self.kind, (date, datetime))
        except TypeError:
            return False

    def promote_with(self, value: Any) -> "DataType":
        """
        Promote this DataType to accommodate a new Python value.
        
        Never mutates; always returns new DataType.
        
        Parameters
        ----------
        value : Any
            Python scalar to accommodate
            
        Returns
        -------
        DataType
            New (possibly promoted) DataType
        """
        # Case 1: None just lifts nullability
        if value is None:
            if self.nullable:
                return self
            return DataType(self.kind, nullable=True)

        vtype = type(value)

        # Case 2: Exact match
        if vtype is self.kind:
            return self

        # Case 3: Numeric ladder (bool → int → float → complex)
        if self.is_numeric and isinstance(value, (int, float, complex, bool)):
            if self.kind is complex or vtype is complex:
                new_kind = complex
            elif self.kind is float or vtype is float:
                new_kind = float
            elif self.kind is int or vtype is int:
                new_kind = int
            else:
                new_kind = bool
            
            if new_kind != self.kind:
                return DataType(new_kind, self.nullable)
            return self

        # Case 4: Temporal ladder (date → datetime)
        if self.is_temporal and isinstance(value, (date, datetime)):
            if self.kind is datetime or vtype is datetime:
                new_kind = datetime
            else:
                new_kind = date
            
            if new_kind != self.kind:
                return DataType(new_kind, self.nullable)
            return self

        # Case 5: String/bytes stay as-is for same type
        if self.kind in (str, bytes) and vtype is self.kind:
            return self

        # Case 6: Degrade to object
        if self.kind is not object:
            warnings.warn(
                f"Degrading column<{self.kind.__name__}> to column<object> "
                f"due to incompatible value of type {vtype.__name__}",
                stacklevel=3,
            )
            return DataType(object, self.nullable)

        # Already object — trivial
        return self


def infer_kind(value: Any) -> Optional[Type]:
    """
    Infer Python type for a single scalar.
    
    Returns None for None values.
    """
    if value is None:
        return None
    
    # Check bool BEFORE int (bool is subclass of int)
    if isinstance(value, bool):
        return bool
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    if isinstance(value, complex):
        return complex
    if isinstance(value, str):
        return str
    if isinstance(value, bytes):
        return bytes
    
    # Check datetime BEFORE date (datetime is subclass of date)
    if isinstance(value, datetime):
        return datetime
    if isinstance(value, date):
        return date
    
    if isinstance(value, list):
        return list
    if isinstance(value, dict):
        return dict
    if isinstance(value, tuple):
        return tuple
    
    return object


def infer_dtype(values: Iterable[Any]) -> DataType:
    """
    Infer a DataType from an iterable of Python scalars.
    
    Applies promotion across all values.
    
    Parameters
    ----------
    values : Iterable[Any]
        Python scalars to analyze
        
    Returns
    -------
    DataType
        Inferred dtype
        
    Examples
    --------
    >>> infer_dtype([1, 2, 3])
    <int>
    >>> infer_dtype([1, 2.5, 3])
    <float>
    >>> infer_dtype([1, None, 3])
    <int nullable>
    >>> infer_dtype([1, "hello"])
    <object>
    """
    dtype: Optional[DataType] = None

    for v in values:
        if dtype is None:
            # First element
            k = infer_kind(v)
            if k is None:
                dtype = DataType(object, nullable=True)
            else:
                dtype = DataType(k, nullable=False)
        else:
            dtype = dtype.promote_with(v)

    # If all values were None or empty iterable
    if dtype is None:
        return DataType(object, nullable=True)

    return dtype


def validate_scalar(value: Any, dtype: DataType) -> Any:
    """
    Validate (and possibly coerce) a scalar before writing into a vector.
    
    Parameters
    ----------
    value : Any
        Scalar to validate
    dtype : DataType
        Target dtype
        
    Returns
    -------
    Any
        Validated/coerced scalar
        
    Raises
    ------
    TypeError
        If value is incompatible with dtype
    """
    if value is None:
        if not dtype.nullable:
            raise TypeError(
                f"Cannot store None in non-nullable {dtype.kind.__name__} column"
            )
        return None

    vtype = type(value)

    # Exact match
    if vtype is dtype.kind:
        return value

    # Numeric coercions
    if dtype.kind is float and vtype in (int, bool):
        return float(value)
    if dtype.kind is int and vtype is bool:
        return int(value)
    if dtype.kind is complex and vtype in (int, float, bool):
        return complex(value)

    # Temporal promotion
    if dtype.kind is datetime and vtype is date:
        return datetime.combine(value, datetime.min.time())

    # Otherwise incompatible
    raise TypeError(
        f"Incompatible value {value!r} for column<{dtype.kind.__name__}>"
    )


# ============================================================
# BACKWARDS COMPATIBILITY WITH OLD DType STRING-BASED SYSTEM
# ============================================================

# Legacy DType constructor for migration
def DType(kind_str: str, nullable: bool = False, default=None) -> DataType:
    """
    Legacy constructor for old DType("int") calls.
    
    Deprecated: Use DataType(int) instead.
    """
    mapping = {
        "int": int,
        "float": float,
        "bool": bool,
        "string": str,
        "bytes": bytes,
        "date": date,
        "datetime": datetime,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "object": object,
        "complex": complex,
    }
    return DataType(mapping.get(kind_str, object), nullable)


# Legacy function names for compatibility
def infer_kind_from_value(v):
    """Legacy name. Use infer_kind() instead."""
    k = infer_kind(v)
    if k is None:
        return None
    # Map back to string for old code
    mapping = {
        int: "int", float: "float", bool: "bool",
        str: "string", bytes: "bytes",
        date: "date", datetime: "datetime",
        list: "list", dict: "dict", tuple: "tuple",
        complex: "complex",
    }
    return mapping.get(k, "object")


def infer_dtype_from_values(values):
    """Legacy name. Use infer_dtype() instead."""
    return infer_dtype(values)


def validate_scalar_for_dtype(value, dtype):
    """Legacy name. Use validate_scalar() instead."""
    return validate_scalar(value, dtype)


def dtype_from_python_type(py_type, nullable=False, default=None):
    """Convert Python type to DataType."""
    return DataType(py_type, nullable)


def promote_dtypes(a: DataType, b: DataType) -> DataType:
    """
    Legacy function for promoting two DataType instances.
    
    Use a.promote_with(value) instead for cleaner semantics.
    """
    # Promote b's kind into a
    if a.kind == b.kind:
        return DataType(a.kind, a.nullable or b.nullable)
    
    # Use promote_with logic by creating a dummy value
    # This is hacky but maintains backwards compat
    dummy_map = {
        int: 0, float: 0.0, bool: False, str: "", bytes: b"",
        date: date.today(), datetime: datetime.now(),
        list: [], dict: {}, tuple: (), object: object(),
        complex: 0j,
    }
    
    dummy = dummy_map.get(b.kind, object())
    result = a.promote_with(dummy)
    
    # OR the nullable flags
    return DataType(result.kind, a.nullable or b.nullable)


# Legacy Python type → DType mapping (for old code that checks this)
PYTHON_TYPE_TO_DTYPE = {
    int: DataType(int),
    float: DataType(float),
    str: DataType(str),
    bool: DataType(bool),
    bytes: DataType(bytes),
    date: DataType(date),
    datetime: DataType(datetime),
    list: DataType(list),
    dict: DataType(dict),
    tuple: DataType(tuple),
}
