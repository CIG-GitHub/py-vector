# backends/fixed_width.py

from __future__ import annotations
from typing import Optional, Callable
import struct

class FixedWidthBufferBackend:
    """
    A strict, Arrow-like backend for numeric fixed-width types.

    Storage layout:
      - _buf: contiguous bytes (bytearray or memoryview)
      - _mask: bitmask for nulls (bytearray or memoryview)
      - _offset: element offset into the buffer
      - _length: number of elements
      - _format: struct format (e.g., 'i' for int32, 'd' for float64)
      - _width: number of bytes per element
    """

    # --- 1. Construction -------------------------------------------------

    def __init__(
        self,
        buf,
        mask,
        offset: int,
        length: int,
        fmt: str
    ):
        self._buf = buf
        self._mask = mask
        self._offset = offset
        self._length = length
        self._format = fmt
        self._width = struct.calcsize(fmt)

    @classmethod
    def from_python(cls, data, fmt="d"):
        """Construct from a Python iterable of scalars."""
        width = struct.calcsize(fmt)

        # allocate contiguous byte buffer
        buf = bytearray(len(data) * width)
        mv = memoryview(buf)

        # mask: 1 bit per element
        mask_bytes = (len(data) + 7) // 8
        mask = bytearray(mask_bytes)

        for i, x in enumerate(data):
            if x is None:
                # null → mask bit = 0
                continue
            else:
                # non-null → mask bit = 1
                mask[i >> 3] |= (1 << (i & 7))
                # write value
                struct.pack_into(fmt, mv, i * width, x)

        return cls(memoryview(buf), memoryview(mask), 0, len(data), fmt)

    # --- 2. Introspection ------------------------------------------------

    def length(self):
        return self._length

    def get_item(self, idx):
        """Return Python scalar or None."""
        if not self._mask_bit(idx):
            return None
        pos = (self._offset + idx) * self._width
        return struct.unpack_from(self._format, self._buf, pos)[0]

    def to_python(self):
        return [self.get_item(i) for i in range(self._length)]

    # --- internal mask helpers ------------------------------------------

    def _mask_bit(self, idx):
        byte = self._mask[(self._offset + idx) >> 3]
        return (byte >> ((self._offset + idx) & 7)) & 1

    # --- 3. Slicing / Masking -------------------------------------------

    def slice(self, slc: slice):
        start, stop, step = slc.indices(self._length)
        if step != 1:
            # Force fallback for now → returns lists, not views
            return type(self).from_python(self.to_python()[slc], self._format)

        return FixedWidthBufferBackend(
            buf=self._buf,
            mask=self._mask,
            offset=self._offset + start,
            length=stop - start,
            fmt=self._format,
        )

    def take(self, indices):
        """Fancy indexing → must copy."""
        out = []
        for i in indices:
            out.append(self.get_item(i))
        return type(self).from_python(out, fmt=self._format)

    # --- 4. Elementwise ops ---------------------------------------------

    def map_unary(self, func):
        out = []
        for i in range(self._length):
            x = self.get_item(i)
            out.append(func(x) if x is not None else None)
        return type(self).from_python(out, self._format)

    def map_binary(self, other, func):
        if not isinstance(other, FixedWidthBufferBackend):
            # fallback: dispatch to other backend
            py = [func(self.get_item(i), other.get_item(i))
                  for i in range(self._length)]
            return type(self).from_python(py, self._format)

        if self._format != other._format:
            # user can implement type promotion here
            raise TypeError("Format mismatch between fixed-width buffers")

        out = []
        for i in range(self._length):
            x = self.get_item(i)
            y = other.get_item(i)
            if x is None or y is None:
                out.append(None)
            else:
                out.append(func(x, y))
        return type(self).from_python(out, self._format)

    # --- 5. Reductions ---------------------------------------------------

    def reduce(self, func, initial):
        acc = initial
        for i in range(self._length):
            x = self.get_item(i)
            if x is not None:
                acc = func(acc, x)
        return acc

    # --- 6. Concatenation -----------------------------------------------

    def concat(self, other):
        if not isinstance(other, FixedWidthBufferBackend):
            return type(self).from_python(
                self.to_python() + other.to_python(), self._format
            )

        if self._format != other._format:
            raise TypeError("Format mismatch for concat")

        # copy buffers (Arrow would do zero-copy chunked arrays)
        return type(self).from_python(
            self.to_python() + other.to_python(),
            self._format
        )
