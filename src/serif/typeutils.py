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

