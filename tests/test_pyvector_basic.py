"""Basic PyVector operations - creation, length, iteration, copy"""
import pytest
from py_vector import PyVector


class TestCreation:
    """Test basic vector creation"""
    
    @pytest.mark.parametrize("initial,expected_len,expected_dtype", [
        ([1, 2, 3], 3, int),
        ([1.5, 2.5, 3.5], 3, float),
        (['a', 'b', 'c'], 3, str),
        ([], 0, None),
        ([1], 1, int),
    ])
    def test_creation_from_list(self, initial, expected_len, expected_dtype):
        v = PyVector(initial)
        assert len(v) == expected_len
        if expected_dtype is None:
            assert v.schema() is None
        else:
            assert v.schema().kind == expected_dtype
        assert list(v) == initial
    
    def test_creation_empty(self):
        v = PyVector()
        assert len(v) == 0
        assert not v  # Empty vector is falsy
    
    def test_creation_with_name(self):
        v = PyVector([1, 2, 3], name='test_vector')
        assert v.name == 'test_vector'
    
    def test_creation_typesafe(self):
        v = PyVector([1, 2, 3], dtype=int)
        assert not v.schema().nullable
        assert v.schema().kind == int


class TestIteration:
    """Test iteration and access"""
    
    def test_iteration(self):
        v = PyVector([1, 2, 3, 4, 5])
        result = []
        for x in v:
            result.append(x)
        assert result == [1, 2, 3, 4, 5]
    
    def test_list_conversion(self):
        v = PyVector([1, 2, 3])
        assert list(v) == [1, 2, 3]
    
    def test_len(self):
        assert len(PyVector([1, 2, 3])) == 3
        assert len(PyVector([])) == 0


class TestCopy:
    """Test copy behavior - tests behavior, not internals"""
    
    def test_copy_basic(self):
        v1 = PyVector([1, 2, 3])
        v2 = v1.copy()
        assert list(v1) == list(v2)
        assert v1 is not v2
    
    def test_copy_mutation_independence(self):
        """Mutations to copy don't affect original (copy-on-write)"""
        v1 = PyVector([1, 2, 3])
        v2 = v1.copy()
        v2[0] = 999
        assert v1[0] == 1  # Original unchanged
        assert v2[0] == 999  # Copy modified


class TestTypePromotion:
    """Test automatic type promotion"""
    
    def test_int_to_float_promotion(self):
        v = PyVector([1, 2, 3.5])
        assert v.schema().kind == float
        assert list(v) == [1.0, 2.0, 3.5]
    
    def test_no_promotion_with_dtype_int(self):
        # If dtype=int specified, should not promote to float
        v = PyVector([1, 2, 3], dtype=int)
        assert v.schema().kind == int


class TestBooleanBehavior:
    """Test truthiness of vectors"""
    
    def test_empty_vector_is_falsy(self):
        v = PyVector([])
        assert not v
    
    def test_nonempty_vector_is_truthy(self):
        v = PyVector([1])
        assert v
        
    def test_nonempty_typed_vector_is_truthy(self):
        v = PyVector([0], dtype=int)
        assert v  # Even with 0, the vector itself is truthy


class TestFingerprint:
    """Test fingerprint computation"""
    
    def test_fingerprint_exists(self):
        v = PyVector([1, 2, 3])
        fp = v.fingerprint()
        assert isinstance(fp, int)
    
    def test_fingerprint_changes_on_mutation(self):
        v = PyVector([1, 2, 3])
        fp1 = v.fingerprint()
        v[0] = 999
        fp2 = v.fingerprint()
        assert fp1 != fp2
    
    def test_fingerprint_same_for_equal_vectors(self):
        v1 = PyVector([1, 2, 3])
        v2 = PyVector([1, 2, 3])
        # Note: fingerprints should be same for same data
        assert v1.fingerprint() == v2.fingerprint()
