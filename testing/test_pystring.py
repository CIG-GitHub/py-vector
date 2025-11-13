"""_PyString specific tests - string method proxying"""
import pytest
from py_vector import PyVector, _PyString


class TestStringCreation:
    """Test _PyString automatic creation"""
    
    def test_auto_creates_pystring(self):
        v = PyVector(['hello', 'world'])
        assert isinstance(v, _PyString)
        assert v._dtype == str


class TestStringMethods:
    """Test string method proxying"""
    
    def test_upper(self):
        v = PyVector(['hello', 'world'])
        result = v.upper()
        assert isinstance(result, PyVector)
        assert list(result) == ['HELLO', 'WORLD']
    
    def test_lower(self):
        v = PyVector(['HELLO', 'WORLD'])
        result = v.lower()
        assert list(result) == ['hello', 'world']
    
    def test_capitalize(self):
        v = PyVector(['hello', 'world'])
        result = v.capitalize()
        assert list(result) == ['Hello', 'World']
    
    def test_strip(self):
        v = PyVector(['  hello  ', '  world  '])
        result = v.strip()
        assert list(result) == ['hello', 'world']
    
    def test_lstrip(self):
        v = PyVector(['  hello', '  world'])
        result = v.lstrip()
        assert list(result) == ['hello', 'world']
    
    def test_rstrip(self):
        v = PyVector(['hello  ', 'world  '])
        result = v.rstrip()
        assert list(result) == ['hello', 'world']
    
    def test_split(self):
        v = PyVector(['hello world', 'foo bar'])
        result = v.split()
        assert isinstance(result, PyVector)
        assert result[0] == ['hello', 'world']
        assert result[1] == ['foo', 'bar']
    
    def test_replace(self):
        v = PyVector(['hello world', 'hello python'])
        result = v.replace('hello', 'hi')
        assert list(result) == ['hi world', 'hi python']
    
    def test_startswith(self):
        v = PyVector(['hello', 'world', 'help'])
        result = v.startswith('he')
        assert list(result) == [True, False, True]
    
    def test_endswith(self):
        v = PyVector(['hello', 'world', 'help'])
        result = v.endswith('ld')
        assert list(result) == [False, True, False]
    
    def test_find(self):
        v = PyVector(['hello', 'world'])
        result = v.find('o')
        assert list(result) == [4, 1]
    
    def test_count(self):
        v = PyVector(['hello', 'mississippi'])
        result = v.count('l')
        assert list(result) == [2, 0]
    
    def test_isalpha(self):
        v = PyVector(['hello', 'world123', 'python'])
        result = v.isalpha()
        assert list(result) == [True, False, True]
    
    def test_isdigit(self):
        v = PyVector(['123', 'abc', '456'])
        result = v.isdigit()
        assert list(result) == [True, False, True]
    
    def test_isalnum(self):
        v = PyVector(['hello123', 'world!', 'python'])
        result = v.isalnum()
        assert list(result) == [True, False, True]
    
    def test_islower(self):
        v = PyVector(['hello', 'WORLD', 'Python'])
        result = v.islower()
        assert list(result) == [True, False, False]
    
    def test_isupper(self):
        v = PyVector(['HELLO', 'world', 'Python'])
        result = v.isupper()
        assert list(result) == [True, False, False]


class TestStringOperations:
    """Test operations on string vectors"""
    
    def test_add_strings(self):
        v1 = PyVector(['hello', 'good'])
        v2 = PyVector([' world', 'bye'])
        result = v1 + v2
        assert list(result) == ['hello world', 'goodbye']
    
    def test_multiply_string(self):
        v = PyVector(['a', 'b'])
        result = v * 3
        assert list(result) == ['aaa', 'bbb']
    
    def test_comparison_strings(self):
        v = PyVector(['apple', 'banana', 'cherry'])
        result = v == 'banana'
        assert list(result) == [False, True, False]


class TestStringFormatting:
    """Test string formatting methods"""
    
    def test_ljust(self):
        v = PyVector(['a', 'bb', 'ccc'])
        result = v.ljust(5)
        assert list(result) == ['a    ', 'bb   ', 'ccc  ']
    
    def test_rjust(self):
        v = PyVector(['a', 'bb', 'ccc'])
        result = v.rjust(5)
        assert list(result) == ['    a', '   bb', '  ccc']
    
    def test_center(self):
        v = PyVector(['a', 'bb'])
        result = v.center(5)
        assert len(result[0]) == 5
        assert len(result[1]) == 5
    
    def test_zfill(self):
        v = PyVector(['42', '123'])
        result = v.zfill(5)
        assert list(result) == ['00042', '00123']
