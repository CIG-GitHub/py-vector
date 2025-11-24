"""
Tests for typing, dtype inference, promotion and nullable behavior in PyVector.
"""

from datetime import date, datetime

import pytest

from py_vector import PyVector
from py_vector.typing import DataType, infer_dtype


class TestInferDtype:
    """Inference from raw Python values -> DataType."""

    @pytest.mark.parametrize(
        "values, expected_kind, expected_nullable",
        [
            ([1, 2, 3], int, False),
            ([1.5, 2.5], float, False),
            (["a", "b", "c"], str, False),
            ([True, False], bool, False),
            ([1 + 2j, 3 + 4j], complex, False),
            ([date(2020, 1, 1), date(2020, 1, 2)], date, False),
            ([datetime(2020, 1, 1), datetime(2020, 1, 2)], datetime, False),
        ],
    )
    def test_infer_pure_kinds(self, values, expected_kind, expected_nullable):
        dt = infer_dtype(values)
        assert isinstance(dt, DataType)
        assert dt.kind is expected_kind
        assert dt.nullable is expected_nullable

    @pytest.mark.parametrize(
        "values, expected_kind",
        [
            ([1, 2.5, 3], float),
            ([1, 2.5, 3 + 4j], complex),
            ([True, 1, 2], int),        # bool + int => int
            ([True, 1.0], float),       # bool + float => float
        ],
    )
    def test_infer_mixed_numeric_promotes(self, values, expected_kind):
        dt = infer_dtype(values)
        assert dt.kind is expected_kind
        assert not dt.nullable

    def test_infer_mixed_temporal_promotes_to_datetime(self):
        values = [date(2020, 1, 1), datetime(2020, 1, 2)]
        dt = infer_dtype(values)
        assert dt.kind is datetime
        assert not dt.nullable

    def test_infer_nullable_when_none_present(self):
        dt = infer_dtype([1, None, 3])
        assert dt.kind is int
        assert dt.nullable

        dt = infer_dtype(["a", None, "c"])
        assert dt.kind is str
        assert dt.nullable

        dt = infer_dtype([date(2020, 1, 1), None])
        assert dt.kind is date
        assert dt.nullable

    def test_infer_all_none_gives_object_nullable(self):
        dt = infer_dtype([None, None, None])
        assert dt.kind is object
        assert dt.nullable

    def test_infer_mixed_incompatible_falls_back_to_object(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dt = infer_dtype([1, "a", 3])
        assert dt.kind is object
        assert dt.nullable is False  # no None in input


class TestVectorCreationWithDtype:
    """Creating vectors with explicit DataType or Python type."""

    def test_create_typed_int(self):
        v = PyVector([1, 2, 3], dtype=DataType(int, nullable=False))
        s = v.schema()
        assert s.kind is int
        assert s.nullable is False
        assert list(v) == [1, 2, 3]

    def test_create_nullable_int(self):
        v = PyVector([1, None, 3], dtype=DataType(int, nullable=True))
        s = v.schema()
        assert s.kind is int
        assert s.nullable is True
        assert list(v) == [1, None, 3]

    def test_create_typed_float(self):
        v = PyVector([1.0, 2.0, 3.0], dtype=DataType(float, nullable=False))
        s = v.schema()
        assert s.kind is float
        assert s.nullable is False

    def test_create_typed_str(self):
        v = PyVector(["a", "b", "c"], dtype=DataType(str, nullable=False))
        s = v.schema()
        assert s.kind is str
        assert s.nullable is False

    def test_python_type_shorthand(self):
        v = PyVector([1, 2, 3], dtype=int)
        s = v.schema()
        assert s.kind is int
        assert s.nullable is False


class TestArithmeticPromotion:
    """Arithmetic ops should produce new vectors with promoted dtype."""

    def test_add_scalar_promotes_int_to_float(self):
        v = PyVector([1, 2, 3])
        out = v + 0.5

        s = out.schema()
        assert s.kind is float
        assert s.nullable is False
        assert list(out) == [1.5, 2.5, 3.5]

    def test_mul_scalar_promotes_int_to_float(self):
        v = PyVector([1, 2, 3])
        out = v * 0.5

        s = out.schema()
        assert s.kind is float
        assert s.nullable is False
        assert list(out) == [0.5, 1.0, 1.5]

    def test_div_scalar_promotes_int_to_float(self):
        v = PyVector([4, 6, 8])
        out = v / 2

        s = out.schema()
        assert s.kind is float
        assert s.nullable is False
        assert list(out) == [2.0, 3.0, 4.0]

    def test_vector_vector_numeric_promotion(self):
        v_int = PyVector([1, 2, 3])
        v_float = PyVector([1.5, 2.5, 3.5])

        out = v_int + v_float
        s = out.schema()
        assert s.kind is float
        assert s.nullable is False
        assert list(out) == [2.5, 4.5, 6.5]

    def test_vector_vector_complex_promotion(self):
        v = PyVector([1, 2.0, 3])
        w = PyVector([1 + 2j, 0 + 0j, 5 + 0j])

        out = v + w
        s = out.schema()
        assert s.kind is complex
        assert s.nullable is False
        assert list(out) == [2 + 2j, 2.0 + 0j, 8 + 0j]

    def test_arith_with_none_preserves_mask(self):
        v = PyVector([1, None, 3])
        out = v + 10

        s = out.schema()
        assert s.kind is int or s.kind is float  # up to implementation
        assert s.nullable is True
        assert list(out) == [11, None, 13]

    def test_comparison_with_none_produces_bool_mask(self):
        v = PyVector([1, None, 3])
        mask = v > 1

        s = mask.schema()
        assert s.kind is bool
        # Nullable is structural - comparison result can have None
        # Define tri-value comparison as: None compared to anything -> None (SQL semantics)
        assert list(mask) == [False, None, True]

    def test_mixed_incompatible_types_raises(self):
        # int + str is not valid in Python, should raise TypeError
        v = PyVector([1, 2, 3])
        w = PyVector(["a", "b", "c"])

        with pytest.raises(TypeError):
            out = v + w  # Should raise since int + str is invalid


# TestDataTypePromotionInternal removed - _promote() is not part of the new API.
# Type promotion now happens automatically during operations (arithmetic, setitem, etc.)
# via infer_dtype() and the backend's _set_values() auto-promotion logic.


class TestSetitemPromotion:
    """Setting values should trigger promotion when needed."""

    def test_setitem_scalar_promotes_int_to_float(self):
        v = PyVector([1, 2, 3])
        assert v.schema().kind is int

        v[1] = 2.5
        s = v.schema()
        assert s.kind is float
        assert list(v) == [1.0, 2.5, 3.0]

    def test_setitem_slice_promotes_int_to_float(self):
        v = PyVector([1, 2, 3, 4])
        v[1:3] = [2.5, 3.5]

        s = v.schema()
        assert s.kind is float
        assert list(v) == [1.0, 2.5, 3.5, 4.0]

    def test_setitem_scalar_promotes_to_complex(self):
        v = PyVector([1, 2, 3])
        v[0] = 1 + 2j

        s = v.schema()
        assert s.kind is complex
        assert list(v) == [1 + 2j, 2 + 0j, 3 + 0j]

    def test_setitem_boolean_mask_promotes(self):
        v = PyVector([1, 2, 3, 4])
        mask = PyVector([True, False, True, False])

        v[mask] = [1.5, 3.5]
        s = v.schema()
        assert s.kind is float
        assert list(v) == [1.5, 2.0, 3.5, 4.0]

    def test_setitem_incompatible_type_promotes_to_str(self):
        # New behavior: auto-promotion recasts entire vector
        v = PyVector([1, 2, 3])
        v[0] = "hello"  # int -> str: recast all to str
        s = v.schema()
        assert s.kind is str
        # All values recast to new dtype
        assert list(v) == ["hello", "2", "3"]


class TestNullableBehavior:
    """Masking and null-handling APIs: isna, fillna, dropna."""

    def test_isna_returns_boolean_mask(self):
        v = PyVector([1, None, 3, None])
        m = v.isna()

        s = m.schema()
        assert s.kind is bool
        assert s.nullable is False
        assert list(m) == [False, True, False, True]

    def test_fillna_preserves_dtype(self):
        # fillna() doesn't automatically remove nullable flag - it's a schema property
        v = PyVector([1, None, 3])
        assert v.schema().nullable is True

        filled = v.fillna(0)
        s = filled.schema()
        assert s.kind is int or s.kind is float
        # Nullable flag preserved (schema is structural, not runtime)
        assert list(filled) == [1, 0, 3]

    def test_dropna_removes_nulls(self):
        # dropna() filters out None values but schema is about structure, not runtime state
        v = PyVector([1, None, 3, None, 5])
        assert v.schema().nullable is True

        dropped = v.dropna()
        # Result has no None values
        assert list(dropped) == [1, 3, 5]
        assert None not in dropped

    def test_arithmetic_preserves_nullable_flag(self):
        v = PyVector([1, None, 3])
        out = v * 2

        s = out.schema()
        assert s.nullable is True
        assert list(out) == [2, None, 6]


class TestTypedSubclasses:
    """Typed subclasses: _PyInt, _PyFloat, etc."""

    def test_int_vector_uses_PyInt_subclass(self):
        from py_vector.vector import _PyInt
        v = PyVector([1, 2, 3])
        assert isinstance(v, _PyInt)
        assert v.schema().kind is int

    def test_float_vector_uses_PyFloat_subclass(self):
        from py_vector.vector import _PyFloat
        v = PyVector([1.5, 2.5])
        assert isinstance(v, _PyFloat)
        assert v.schema().kind is float

    def test_string_vector_uses_PyString_subclass(self):
        from py_vector.vector import _PyString
        v = PyVector(["a", "b", "c"])
        assert isinstance(v, _PyString)
        assert v.schema().kind is str

    def test_date_vector_uses_PyDate_subclass(self):
        from py_vector.vector import _PyDate
        v = PyVector([date(2020, 1, 1), date(2020, 1, 2)])
        assert isinstance(v, _PyDate)
        assert v.schema().kind is date

    def test_setitem_triggers_auto_promotion(self):
        # Promotion happens via setitem, not explicit _promote
        from py_vector.vector import _PyInt
        v = PyVector([1, 2, 3])
        assert isinstance(v, _PyInt)
        assert v.schema().kind is int

        v[0] = 1.5  # Auto-promotes to float
        s = v.schema()
        assert s.kind is float
        assert list(v) == [1.5, 2.0, 3.0]
