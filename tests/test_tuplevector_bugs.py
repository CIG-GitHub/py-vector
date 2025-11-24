"""
Tests for TupleVector backend bug fixes.

This module tests the internal TupleVector backend to ensure
all identified bugs are properly fixed.
"""

import pytest
import math
from datetime import date, datetime

from py_vector.tuplevector import TupleVector, MethodProxy
from py_vector.typing import DataType
from py_vector.errors import PyVectorTypeError, PyVectorValueError


class TestBug01_FingerprintInvalidation:
    """Bug #1: Fingerprint not invalidated on dtype promotion during _set_values."""
    
    def test_fingerprint_invalidated_on_promotion(self):
        """Fingerprint should be recalculated after dtype promotion."""
        v = TupleVector([1, 2, 3], dtype=DataType(int, nullable=False))
        fp_before = v.fingerprint()
        
        # Mutate with incompatible type, forcing promotion
        v._set_values([(1, 2.5)])  # int -> float promotion
        
        # Fingerprint should change
        fp_after = v.fingerprint()
        assert fp_before != fp_after, "Fingerprint should change after dtype promotion"
        
        # Verify dtype was actually promoted
        assert v.schema().kind is float


class TestBug02_NoneSafeMethodProxy:
    """Bug #2: MethodProxy crashes on None elements."""
    
    def test_method_proxy_handles_none(self):
        """Method calls should propagate None instead of crashing."""
        v = TupleVector(["hello", None, "world"])
        result = v.upper()
        
        assert list(result._storage) == ["HELLO", None, "WORLD"]
    
    def test_method_proxy_with_args(self):
        """Method proxy should handle methods with arguments."""
        v = TupleVector(["hello", None, "world"])
        result = v.replace("o", "0")
        
        assert list(result._storage) == ["hell0", None, "w0rld"]
    
    def test_method_proxy_raises_on_missing_method(self):
        """Should raise AttributeError if method doesn't exist on elements."""
        v = TupleVector([1, 2, 3])
        
        with pytest.raises(AttributeError, match="has no method 'nonexistent'"):
            v.nonexistent()


class TestBug07_DatetimeCasting:
    """Bug #7: Casting to date/datetime from ISO strings."""
    
    def test_cast_to_date(self):
        """Should cast ISO date strings to date objects."""
        v = TupleVector(["2020-01-01", "2020-12-31"])
        result = v.cast(date)
        
        assert result._storage[0] == date(2020, 1, 1)
        assert result._storage[1] == date(2020, 12, 31)
    
    def test_cast_to_datetime(self):
        """Should cast ISO datetime strings to datetime objects."""
        v = TupleVector(["2020-01-01T12:00:00", "2020-12-31T23:59:59"])
        result = v.cast(datetime)
        
        assert result._storage[0] == datetime(2020, 1, 1, 12, 0, 0)
        assert result._storage[1] == datetime(2020, 12, 31, 23, 59, 59)
    
    def test_cast_preserves_none(self):
        """Casting should preserve None values."""
        v = TupleVector(["2020-01-01", None, "2020-12-31"])
        result = v.cast(date)
        
        assert result._storage[1] is None


class TestBug11_TruthinessAmbiguity:
    """Bug #11: Boolean truthiness is ambiguous for vectors."""
    
    def test_bool_raises_error(self):
        """Using a vector in boolean context should raise ValueError."""
        v = TupleVector([True, False, True])
        
        with pytest.raises(ValueError, match="truth value.*ambiguous"):
            if v:
                pass
    
    def test_empty_vector_bool_raises(self):
        """Even empty vectors should raise on __bool__."""
        v = TupleVector([])
        
        with pytest.raises(ValueError, match="truth value.*ambiguous"):
            bool(v)


class TestBug12_FillnaDtypeValidation:
    """Bug #12: fillna doesn't validate fill value against dtype."""
    
    def test_fillna_validates_dtype(self):
        """fillna should reject incompatible fill values."""
        v = TupleVector([1, None, 3], dtype=DataType(int, nullable=True))
        
        # This should fail - can't fill int vector with string
        with pytest.raises(PyVectorTypeError, match="incompatible with dtype"):
            v.fillna("hello")
    
    def test_fillna_accepts_compatible_value(self):
        """fillna should accept compatible fill values."""
        v = TupleVector([1, None, 3], dtype=DataType(int, nullable=True))
        result = v.fillna(0)
        
        assert list(result._storage) == [1, 0, 3]


class TestBug13_DeterministicSetHashing:
    """Bug #13: Set hashing is non-deterministic due to random iteration order."""
    
    def test_set_hashing_is_deterministic(self):
        """Sets should hash the same way across multiple runs."""
        v1 = TupleVector([{1, 2, 3}])
        v2 = TupleVector([{3, 2, 1}])  # Same set, different literal order
        
        # Should produce same fingerprint
        assert v1.fingerprint() == v2.fingerprint()
    
    def test_nested_set_hashing(self):
        """Nested structures with sets should be deterministic."""
        v1 = TupleVector([({1, 2}, {3, 4})])
        v2 = TupleVector([({2, 1}, {4, 3})])
        
        assert v1.fingerprint() == v2.fingerprint()
    
    def test_nan_hashing(self):
        """NaN values should hash deterministically."""
        v1 = TupleVector([float('nan'), 1.0, 2.0])
        v2 = TupleVector([float('nan'), 1.0, 2.0])
        
        # NaN fingerprints should match
        assert v1.fingerprint() == v2.fingerprint()


class TestBug14_UniqueWithUnhashable:
    """Bug #14: unique() fails on unhashable elements."""
    
    def test_unique_with_hashable(self):
        """unique() works normally with hashable elements."""
        v = TupleVector([1, 2, 1, 3, 2])
        result = v.unique()
        
        assert list(result._storage) == [1, 2, 3]
    
    def test_unique_with_unhashable_lists(self):
        """unique() should handle unhashable elements like lists."""
        v = TupleVector([[1, 2], [3, 4], [1, 2], [5, 6]])
        result = v.unique()
        
        assert list(result._storage) == [[1, 2], [3, 4], [5, 6]]
    
    def test_unique_with_unhashable_dicts(self):
        """unique() should handle unhashable dicts."""
        v = TupleVector([{'a': 1}, {'b': 2}, {'a': 1}])
        result = v.unique()
        
        assert list(result._storage) == [{'a': 1}, {'b': 2}]
    
    def test_unique_preserves_order(self):
        """unique() should preserve first-appearance order."""
        v = TupleVector([3, 1, 2, 1, 3])
        result = v.unique()
        
        assert list(result._storage) == [3, 1, 2]


class TestBug16_LazyFingerprintComputation:
    """Bug #16: Fingerprint computed eagerly, wasting cycles."""
    
    def test_fingerprint_computed_lazily(self):
        """Fingerprint should not be computed until requested."""
        v = TupleVector([1, 2, 3, 4, 5])
        
        # Internal state should show no fingerprint yet
        assert v._fp is None
        assert v._fp_powers is None
        
        # Access fingerprint
        fp = v.fingerprint()
        
        # Now it should be cached
        assert v._fp is not None
        assert v._fp == fp
    
    def test_fingerprint_cached(self):
        """Subsequent fingerprint calls should use cache."""
        v = TupleVector([1, 2, 3])
        
        fp1 = v.fingerprint()
        fp2 = v.fingerprint()
        
        # Should return same cached value
        assert fp1 == fp2


class TestTupleVectorBasics:
    """Basic TupleVector functionality tests."""
    
    def test_initialization(self):
        """TupleVector should initialize correctly."""
        v = TupleVector([1, 2, 3], dtype=DataType(int), name="test")
        
        assert len(v) == 3
        assert v._name == "test"
        assert v.schema().kind is int
    
    def test_indexing(self):
        """Should support integer indexing."""
        v = TupleVector([10, 20, 30])
        
        assert v[0] == 10
        assert v[1] == 20
        assert v[2] == 30
    
    def test_slicing(self):
        """Should support slicing."""
        v = TupleVector([1, 2, 3, 4, 5])
        result = v[1:4]
        
        assert list(result._storage) == [2, 3, 4]
        assert isinstance(result, TupleVector)
    
    def test_clone_with(self):
        """clone_with should preserve metadata."""
        v = TupleVector([1, 2, 3], dtype=DataType(int), name="original")
        cloned = v.clone_with([4, 5, 6])
        
        assert list(cloned._storage) == [4, 5, 6]
        assert cloned._name == "original"
        assert cloned.schema().kind is int


class TestTupleVectorOperations:
    """Test arithmetic and reduction operations."""
    
    def test_elementwise_with_vector(self):
        """_elementwise should work with another vector."""
        v1 = TupleVector([1, 2, 3])
        v2 = TupleVector([10, 20, 30])
        
        result = v1._elementwise(lambda a, b: a + b, v2)
        assert list(result._storage) == [11, 22, 33]
    
    def test_elementwise_with_scalar(self):
        """_elementwise should work with scalars."""
        v = TupleVector([1, 2, 3])
        
        result = v._elementwise(lambda a, b: a * b, 10)
        assert list(result._storage) == [10, 20, 30]
    
    def test_reduce(self):
        """_reduce should aggregate values."""
        v = TupleVector([1, 2, 3, 4, 5])
        
        total = v._reduce(lambda acc, x: acc + x)
        assert total == 15
    
    def test_reduce_empty_raises(self):
        """_reduce on empty vector should raise."""
        v = TupleVector([])
        
        with pytest.raises(ValueError, match="Cannot reduce empty vector"):
            v._reduce(lambda acc, x: acc + x)


class TestTupleVectorEdgeCases:
    """Edge cases and error handling."""
    
    def test_empty_vector(self):
        """Empty vectors should work."""
        v = TupleVector([])
        
        assert len(v) == 0
        assert v.fingerprint() == 0  # Empty fingerprint
    
    def test_none_only_vector(self):
        """Vector with only None values."""
        v = TupleVector([None, None, None])
        
        assert len(v) == 3
        assert all(x is None for x in v._storage)
    
    def test_mixed_types(self):
        """Vector with mixed types should infer object dtype."""
        v = TupleVector([1, "hello", 3.14])
        
        # Should infer object dtype
        assert v.schema().kind is object
    
    def test_invalid_index_type(self):
        """Invalid index type should raise."""
        v = TupleVector([1, 2, 3])
        
        with pytest.raises(PyVectorTypeError):
            v[{1, 2}]  # Set is not valid index
