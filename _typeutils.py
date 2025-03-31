"""
Utilities from nonsense
"""

def slice_length(s: slice, sequence_length: int) -> int:
    """
    Calculates the length of a slice given the slice object and the sequence length.

    Args:
        s: The slice object.
        sequence_length: The length of the sequence being sliced.

    Returns:
        The length of the slice.
    """
    start, stop, step = s.indices(sequence_length)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)



slices_to_test = [
	slice(2, 7),
	slice(1, -1),
	slice(None, None, 2),
	slice(7, 2, -1),
	slice(-2, -7, -3),
]
for rng in range(90, 110):
	my_list = [x for x in range(rng)]

	for t in slices_to_test:
		print(f"{slice_length(t, len(my_list))==len(my_list[t])}: {slice_length(t, len(my_list))} {len(my_list[t])}")
