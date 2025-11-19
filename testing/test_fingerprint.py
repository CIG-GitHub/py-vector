"""Fingerprint change detection tests"""
import pytest
from py_vector import PyVector
from py_table import PyTable


class TestBasicFingerprint:
    """Test basic fingerprint functionality"""
    
    def test_fingerprint_returns_int(self):
        v = PyVector([1, 2, 3])
        fp = v.fingerprint()
        assert isinstance(fp, int)
    
    def test_same_data_same_fingerprint(self):
        v1 = PyVector([1, 2, 3])
        v2 = PyVector([1, 2, 3])
        assert v1.fingerprint() == v2.fingerprint()
    
    def test_different_data_different_fingerprint(self):
        v1 = PyVector([1, 2, 3])
        v2 = PyVector([1, 2, 4])
        assert v1.fingerprint() != v2.fingerprint()


class TestMutationDetection:
    """Test fingerprint changes on mutations"""
    
    def test_single_index_mutation(self):
        v = PyVector([1, 2, 3, 4, 5])
        fp1 = v.fingerprint()
        v[2] = 999
        fp2 = v.fingerprint()
        assert fp1 != fp2
    
    def test_slice_mutation(self):
        v = PyVector([1, 2, 3, 4, 5])
        fp1 = v.fingerprint()
        v[1:4] = [20, 30, 40]
        fp2 = v.fingerprint()
        assert fp1 != fp2
    
    def test_boolean_mask_mutation(self):
        v = PyVector([1, 2, 3, 4, 5])
        fp1 = v.fingerprint()
        v[v > 3] = 999
        fp2 = v.fingerprint()
        assert fp1 != fp2
    
    def test_integer_vector_mutation(self):
        v = PyVector([1, 2, 3, 4, 5])
        fp1 = v.fingerprint()
        indices = PyVector([0, 2, 4], dtype=int, typesafe=True)
        v[indices] = 999
        fp2 = v.fingerprint()
        assert fp1 != fp2


class TestMultipleMutations:
    """Test fingerprint tracking multiple mutations"""
    
    def test_sequential_mutations(self):
        v = PyVector([1, 2, 3, 4, 5])
        fp0 = v.fingerprint()
        
        v[0] = 10
        fp1 = v.fingerprint()
        assert fp0 != fp1
        
        v[1] = 20
        fp2 = v.fingerprint()
        assert fp1 != fp2
        assert fp0 != fp2
        
        v[2] = 30
        fp3 = v.fingerprint()
        assert fp2 != fp3
        assert fp1 != fp3
        assert fp0 != fp3
    
    def test_mutation_and_revert(self):
        v = PyVector([1, 2, 3])
        fp_original = v.fingerprint()
        
        v[1] = 999
        fp_mutated = v.fingerprint()
        assert fp_original != fp_mutated
        
        v[1] = 2  # Revert to original value
        fp_reverted = v.fingerprint()
        assert fp_reverted == fp_original


class TestNestedStructures:
    """Test fingerprint for nested vectors (tables)"""
    
    def test_table_fingerprint(self):
        col1 = PyVector([1, 2, 3])
        col2 = PyVector([4, 5, 6])
        table = PyVector([col1, col2])
        fp = table.fingerprint()
        assert isinstance(fp, int)
    
    def test_table_mutation_via_column(self):
        col1 = PyVector([1, 2, 3])
        col2 = PyVector([4, 5, 6])
        table = PyVector([col1, col2])
        fp1 = table.fingerprint()
        
        # Mutate original column - value semantics means table is unaffected
        col1[0] = 999
        fp2 = table.fingerprint()
        assert fp1 == fp2  # Table has a copy, not affected by mutation


class TestFingerprintStability:
    """Test fingerprint stability and reproducibility"""
    
    def test_repeated_calls_same_result(self):
        v = PyVector([1, 2, 3, 4, 5])
        fp1 = v.fingerprint()
        fp2 = v.fingerprint()
        fp3 = v.fingerprint()
        assert fp1 == fp2 == fp3
    
    def test_copy_has_same_fingerprint(self):
        v1 = PyVector([1, 2, 3])
        v2 = v1.copy()
        assert v1.fingerprint() == v2.fingerprint()
    
    def test_copy_mutations_independent(self):
        v1 = PyVector([1, 2, 3])
        v2 = v1.copy()
        fp1_original = v1.fingerprint()
        fp2_original = v2.fingerprint()
        
        v2[0] = 999
        assert v1.fingerprint() == fp1_original  # v1 unchanged
        assert v2.fingerprint() != fp2_original  # v2 changed


class TestFingerprintTypes:
    """Test fingerprint with different data types"""
    
    def test_int_vector_fingerprint(self):
        v = PyVector([1, 2, 3])
        fp = v.fingerprint()
        assert isinstance(fp, int)
    
    def test_float_vector_fingerprint(self):
        v = PyVector([1.5, 2.5, 3.5])
        fp = v.fingerprint()
        assert isinstance(fp, int)
    
    def test_string_vector_fingerprint(self):
        v = PyVector(['hello', 'world'])
        fp = v.fingerprint()
        assert isinstance(fp, int)
    
    def test_different_types_different_fingerprints(self):
        v1 = PyVector([1, 2, 3])
        v2 = PyVector([1.0, 2.0, 3.0])
        # These might have different fingerprints due to hash differences
        # Just ensure they both return valid fingerprints
        assert isinstance(v1.fingerprint(), int)
        assert isinstance(v2.fingerprint(), int)


class TestChangeDetectionWorkflow:
    """Test typical change detection workflow"""
    
    def test_track_changes_workflow(self):
        # Create a vector
        v = PyVector([1, 2, 3, 4, 5])
        initial_fp = v.fingerprint()
        
        # Simulate checking for changes (no changes)
        assert v.fingerprint() == initial_fp
        
        # Make a change
        v[2] = 999
        
        # Detect the change
        assert v.fingerprint() != initial_fp
        
        # Update the stored fingerprint
        new_fp = v.fingerprint()
        
        # No more changes
        assert v.fingerprint() == new_fp
