"""
Quick test to verify the tuple -> list migration works
"""

from py_vector import PyVector

# Test 1: Basic creation
print("Test 1: Basic creation")
v = PyVector([1, 2, 3, 4, 5])
print(f"  Created: {v._underlying}")
print(f"  Type: {type(v._underlying)}")
assert isinstance(v._underlying, list), "Should be a list!"
print("  ✓ Pass\n")

# Test 2: In-place mutation
print("Test 2: In-place mutation")
v[0] = 99
print(f"  After v[0] = 99: {v._underlying}")
assert v[0] == 99, "Mutation should work"
print("  ✓ Pass\n")

# Test 3: Slice assignment
print("Test 3: Slice assignment")
v[1:3] = [20, 30]
print(f"  After v[1:3] = [20, 30]: {v._underlying}")
assert v[1] == 20 and v[2] == 30, "Slice assignment should work"
print("  ✓ Pass\n")

# Test 4: Boolean masking assignment
print("Test 4: Boolean masking")
mask = PyVector([True, False, True, False, False], typesafe=True)
v[mask] = [100, 300]
print(f"  After v[mask] = [100, 300]: {v._underlying}")
assert v[0] == 100 and v[2] == 300, "Boolean masking should work"
print("  ✓ Pass\n")

# Test 5: Operations return new vectors
print("Test 5: Operations return new vectors")
v2 = v + 10
print(f"  v + 10: {v2._underlying}")
print(f"  Original v: {v._underlying}")
assert isinstance(v2._underlying, list), "Result should be a list"
assert v[0] == 100, "Original shouldn't change"
print("  ✓ Pass\n")

# Test 6: Copy
print("Test 6: Copy creates new list")
v3 = v.copy()
v3[0] = 999
print(f"  Copy modified: {v3._underlying}")
print(f"  Original: {v._underlying}")
assert v[0] == 100 and v3[0] == 999, "Copy should be independent"
print("  ✓ Pass\n")

# Test 7: Concatenation
print("Test 7: Concatenation")
v4 = PyVector([1, 2])
v5 = PyVector([3, 4])
v6 = v4 << v5
print(f"  [1,2] << [3,4]: {v6._underlying}")
assert len(v6) == 4 and v6[2] == 3, "Concatenation should work"
print("  ✓ Pass\n")

print("=" * 50)
print("All tests passed! ✓")
print("Lists are working as expected.")
print("=" * 50)
