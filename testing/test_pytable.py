"""PyTable specific tests - 2D operations, column access, matrix operations"""
import pytest
from py_vector import PyVector, PyTable


class TestTableCreation:
    """Test PyTable creation"""
    
    def test_create_from_vectors(self):
        col1 = PyVector([1, 2, 3])
        col2 = PyVector([4, 5, 6])
        table = PyVector([col1, col2])
        assert isinstance(table, PyTable)
        assert table.size() == (3, 2)
    
    def test_create_from_unequal_vectors_warns(self):
        col1 = PyVector([1, 2, 3])
        col2 = PyVector([4, 5])
        with pytest.warns(UserWarning):
            table = PyVector([col1, col2])
    
    def test_empty_table(self):
        col1 = PyVector([])
        col2 = PyVector([])
        table = PyVector([col1, col2])
        assert len(table) == 0


class TestTableSize:
    """Test table dimensions"""
    
    def test_size_2d(self):
        col1 = PyVector([1, 2, 3])
        col2 = PyVector([4, 5, 6])
        table = PyVector([col1, col2])
        assert table.size() == (3, 2)
    
    def test_len_returns_rows(self):
        col1 = PyVector([1, 2, 3])
        col2 = PyVector([4, 5, 6])
        table = PyVector([col1, col2])
        assert len(table) == 3


class TestColumnAccess:
    """Test accessing columns"""
    
    def test_cols_returns_all_columns(self):
        col1 = PyVector([1, 2, 3])
        col2 = PyVector([4, 5, 6])
        table = PyVector([col1, col2])
        cols = table.cols()
        assert len(cols) == 2
        assert list(cols[0]) == [1, 2, 3]
        assert list(cols[1]) == [4, 5, 6]
    
    def test_cols_with_names(self):
        col1 = PyVector([1, 2, 3], name='a')
        col2 = PyVector([4, 5, 6], name='b')
        table = PyVector([col1, col2])
        assert table.a is col1
        assert table.b is col2


class TestRowAccess:
    """Test accessing rows"""
    
    def test_getitem_single_row(self):
        col1 = PyVector([1, 2, 3])
        col2 = PyVector([4, 5, 6])
        table = PyVector([col1, col2])
        row = table[0]
        assert isinstance(row, PyVector)
        assert list(row) == [1, 4]
    
    def test_getitem_row_slice(self):
        col1 = PyVector([1, 2, 3, 4])
        col2 = PyVector([5, 6, 7, 8])
        table = PyVector([col1, col2])
        result = table[1:3]
        assert result.size() == (2, 2)
        assert list(result.cols()[0]) == [2, 3]
        assert list(result.cols()[1]) == [6, 7]
    
    def test_getitem_boolean_mask(self):
        col1 = PyVector([1, 2, 3, 4])
        col2 = PyVector([5, 6, 7, 8])
        table = PyVector([col1, col2])
        mask = [True, False, True, False]
        result = table[mask]
        assert result.size() == (2, 2)
        assert list(result.cols()[0]) == [1, 3]


class TestTableIndexing:
    """Test 2D indexing"""
    
    def test_2d_indexing_single_element(self):
        col1 = PyVector([1, 2, 3])
        col2 = PyVector([4, 5, 6])
        table = PyVector([col1, col2])
        assert table[0, 0] == 1
        assert table[1, 1] == 5
        assert table[2, 0] == 3
    
    def test_2d_indexing_row_slice(self):
        col1 = PyVector([1, 2, 3, 4])
        col2 = PyVector([5, 6, 7, 8])
        table = PyVector([col1, col2])
        result = table[1:3, 0]
        assert list(result) == [2, 3]


class TestTranspose:
    """Test transpose operation"""
    
    def test_transpose_2d(self):
        col1 = PyVector([1, 2, 3])
        col2 = PyVector([4, 5, 6])
        table = PyVector([col1, col2])
        transposed = table.T
        # After transpose, we get rows as vectors
        # Original: 3 rows × 2 columns → Transposed: 2 rows × 3 columns
        assert transposed.size() == (2, 3)


class TestTableOperations:
    """Test operations on tables"""
    
    def test_sum_table(self):
        col1 = PyVector([1, 2, 3])
        col2 = PyVector([4, 5, 6])
        table = PyVector([col1, col2])
        result = table.sum()
        assert list(result) == [6, 15]
    
    def test_mean_table(self):
        col1 = PyVector([1, 2, 3])
        col2 = PyVector([4, 5, 6])
        table = PyVector([col1, col2])
        result = table.mean()
        assert list(result) == [2.0, 5.0]
    
    def test_max_table(self):
        col1 = PyVector([1, 2, 3])
        col2 = PyVector([4, 5, 6])
        table = PyVector([col1, col2])
        result = table.max()
        assert list(result) == [3, 6]
    
    def test_min_table(self):
        col1 = PyVector([1, 2, 3])
        col2 = PyVector([4, 5, 6])
        table = PyVector([col1, col2])
        result = table.min()
        assert list(result) == [1, 4]


class TestConcatenation:
    """Test concatenating tables"""
    
    def test_rshift_adds_column(self):
        col1 = PyVector([1, 2, 3])
        col2 = PyVector([4, 5, 6])
        col3 = PyVector([7, 8, 9])
        table = PyVector([col1, col2])
        result = table >> col3
        assert result.size() == (3, 3)
    
    def test_lshift_adds_rows(self):
        col1 = PyVector([1, 2, 3])
        col2 = PyVector([4, 5, 6])
        table = PyVector([col1, col2])
        # This should append rows (extend each column)
        new_row_table = PyVector([PyVector([7]), PyVector([8])])
        result = table << new_row_table
        # Each column should be extended
        assert len(result.cols()[0]) == 4
        assert list(result.cols()[0]) == [1, 2, 3, 7]


class TestTableFingerprint:
    """Test fingerprint for tables"""
    
    def test_table_fingerprint_changes_on_column_mutation(self):
        col1 = PyVector([1, 2, 3])
        col2 = PyVector([4, 5, 6])
        table = PyVector([col1, col2])
        fp1 = table.fingerprint()
        col1[0] = 999  # Mutate a column
        fp2 = table.fingerprint()
        assert fp1 != fp2
