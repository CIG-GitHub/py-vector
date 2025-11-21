"""
Centralized DType system for PyVector / PyTable.

Handles:
  - supported kinds
  - nullability
  - dtype inference
  - type promotion
  - mixed-type rules
  - scalar validation

This module is backend-agnostic and forms the semantic core
of PyVector's type system.
"""

from datetime import date, datetime


# Supported type kinds
VALID_KINDS = {
    "int",
    "float",
    "bool",
    "string",
    "date",
    "datetime",
    "bytes",
    "list",
    "dict",
    "tuple",
    "object",   # fallback for heterogeneous or unknown values
}


class DType:
    """
    Represents the semantic type of a PyVector.

    Attributes
    ----------
    kind : str
        Base kind: 'int', 'float', 'string', etc.
    nullable : bool
        Whether the vector contains missing values (None).
    default : object
        Default value for this dtype (used for padding/filling)

    Notes
    -----
    - Nullability is handled at the dtype level, NOT as a separate flag.
    - DType only handles same-kind promotion; mixed kinds use the
      global promotion table.
    
    Examples
    --------
    >>> DType("int")
    <int>
    >>> DType("int", nullable=True)
    <int nullable>
    >>> DType("float", nullable=True, default=0.0)
    <float nullable default=0.0>
    """

    __slots__ = ("kind", "nullable", "default")

    def __init__(self, kind: str, nullable: bool = False, default=None):
        if kind not in VALID_KINDS:
            raise ValueError(f"Unknown dtype kind: {kind}")
        self.kind = kind
        self.nullable = bool(nullable)
        self.default = default

    def __repr__(self):
        parts = [self.kind]
        if self.nullable:
            parts.append("nullable")
        if self.default is not None:
            parts.append(f"default={self.default!r}")
        return f"<{' '.join(parts)}>"

    def __eq__(self, other):
        if not isinstance(other, DType):
            return False
        return (self.kind == other.kind and 
                self.nullable == other.nullable and
                self.default == other.default)

    def promote_same_kind(self, other: "DType") -> "DType":
        """
        Combine two dtypes with the same base kind.
        Nullability is OR'ed, default is taken from self.
        """
        assert self.kind == other.kind
        return DType(
            self.kind, 
            self.nullable or other.nullable,
            self.default if self.default is not None else other.default
        )

    def with_nullable(self, nullable=True):
        """Return a copy with nullable flag set."""
        return DType(self.kind, nullable, self.default)

    def with_default(self, default):
        """Return a copy with default value set."""
        return DType(self.kind, self.nullable, default)


# Mixed-type promotion rules
# When kinds differ, consult this registry
TYPE_PROMOTION_TABLE = {
    ("int", "int"):       "int",
    ("int", "float"):     "float",
    ("float", "int"):     "float",
    ("float", "float"):   "float",

    ("bool", "int"):      "int",
    ("int", "bool"):      "int",
    ("bool", "bool"):     "bool",

    ("string", "string"): "string",
    ("string", "bytes"):  "object",
    ("bytes", "string"):  "object",

    # string mixed with numeric → object
    ("string", "int"):    "object",
    ("int", "string"):    "object",
    ("string", "float"):  "object",
    ("float", "string"):  "object",

    ("bytes", "bytes"):   "bytes",

    # dates only mix with same-kind
    ("date", "date"):         "date",
    ("datetime", "datetime"): "datetime",
    ("date", "datetime"):     "object",
    ("datetime", "date"):     "object",
    
    # collections
    ("list", "list"):     "list",
    ("dict", "dict"):     "dict",
    ("tuple", "tuple"):   "tuple",
}


def promote_kind(a_kind: str, b_kind: str) -> str:
    """Promote two kind strings using TYPE_PROMOTION_TABLE."""
    if a_kind == b_kind:
        return a_kind

    key = (a_kind, b_kind)
    if key in TYPE_PROMOTION_TABLE:
        return TYPE_PROMOTION_TABLE[key]

    # Fallback: anything with object → object
    if a_kind == "object" or b_kind == "object":
        return "object"

    # Unknown combination → object (safe fallback)
    return "object"


def promote_dtypes(a: DType, b: DType) -> DType:
    """
    Promote two DType instances to a resulting DType.

    - If kinds match → use same-kind promotion
    - Otherwise → use global promotion table
    - Nullability OR's together
    """
    if a.kind == b.kind:
        return a.promote_same_kind(b)

    base = promote_kind(a.kind, b.kind)
    nullable = a.nullable or b.nullable
    default = a.default if a.default is not None else b.default
    return DType(base, nullable, default)


def infer_kind_from_value(v):
    """Infer primitive kind for a single Python value."""
    if v is None:
        return None

    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "string"
    if isinstance(v, bytes):
        return "bytes"
    if isinstance(v, datetime):
        return "datetime"
    if isinstance(v, date):
        return "date"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    if isinstance(v, tuple):
        return "tuple"

    return "object"


def infer_dtype_from_values(values):
    """
    Given an iterable of Python values, infer:
      - base kind
      - nullability
    """
    nullable = False
    kinds = set()

    for v in values:
        k = infer_kind_from_value(v)
        if k is None:
            nullable = True
            continue
        kinds.add(k)

    if not kinds:
        # All values were None → use object+nullable
        return DType("object", nullable=True)

    if len(kinds) == 1:
        return DType(kinds.pop(), nullable=nullable)

    # Mixed kinds → promote across them
    kinds_list = list(kinds)
    current = kinds_list[0]
    for k in kinds_list[1:]:
        current = promote_kind(current, k)

    return DType(current, nullable)


def validate_scalar_for_dtype(value, dtype: DType):
    """
    Validate / coerce a Python scalar before inserting into a vector.
    Raises if incompatible.
    """
    if value is None:
        if not dtype.nullable:
            raise TypeError(f"Cannot insert None into non-nullable dtype {dtype}")
        return None

    kind = infer_kind_from_value(value)

    if kind == dtype.kind:
        return value

    # Numeric coercions
    if dtype.kind == "float" and kind == "int":
        return float(value)
    if dtype.kind == "int" and kind == "bool":
        return int(value)

    # Everything else falls back to user error
    raise TypeError(f"Cannot insert value {value!r} into dtype {dtype}")


# Convenience: Python type → DType mapping
PYTHON_TYPE_TO_DTYPE = {
    int: DType("int"),
    float: DType("float"),
    str: DType("string"),
    bool: DType("bool"),
    bytes: DType("bytes"),
    date: DType("date"),
    datetime: DType("datetime"),
    list: DType("list"),
    dict: DType("dict"),
    tuple: DType("tuple"),
}


def dtype_from_python_type(py_type, nullable=False, default=None):
    """Convert Python type to DType (for backwards compatibility)."""
    if py_type in PYTHON_TYPE_TO_DTYPE:
        base = PYTHON_TYPE_TO_DTYPE[py_type]
        return DType(base.kind, nullable=nullable, default=default)
    return DType("object", nullable=nullable, default=default)
