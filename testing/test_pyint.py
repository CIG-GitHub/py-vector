"""_PyInt specific tests - int method proxying"""
import pytest
from py_vector import PyVector, _PyInt


class TestIntCreation:
    """Test _PyInt automatic creation"""
    
    def test_auto_creates_pyint(self):
        v = PyVector([1, 2, 3])
        assert isinstance(v, _PyInt)
        assert v._dtype == int
    
    def test_does_not_include_bool(self):
        # Bools should not create _PyInt
        v = PyVector([True, False, True])
        assert not isinstance(v, _PyInt)


class TestIntMethods:
    """Test int method proxying"""
    
    def test_bit_length(self):
        v = PyVector([1, 2, 4, 8, 16])
        result = v.bit_length()
        assert isinstance(result, PyVector)
        assert list(result) == [1, 2, 3, 4, 5]
    
    def test_bit_count(self):
        v = PyVector([0, 1, 3, 7, 15])  # 0, 1, 11, 111, 1111 in binary
        result = v.bit_count()
        assert isinstance(result, PyVector)
        assert list(result) == [0, 1, 2, 3, 4]
    
    def test_to_bytes(self):
        v = PyVector([1, 255, 256])
        result = v.to_bytes(2, 'big')
        assert isinstance(result, PyVector)
        assert len(result) == 3
        # Just check they're bytes
        for x in result:
            assert isinstance(x, bytes)


class TestIntOperations:
    """Test operations on int vectors"""
    
    def test_floor_division(self):
        v = PyVector([10, 21, 30])
        result = v // 3
        assert isinstance(result, PyVector)
        assert v._dtype == int
        assert list(result) == [3, 7, 10]
    
    def test_modulo(self):
        v = PyVector([10, 21, 30])
        result = v % 7
        assert list(result) == [3, 0, 2]
    
    def test_power_with_ints(self):
        v = PyVector([2, 3, 4])
        result = v ** 2
        assert list(result) == [4, 9, 16]
