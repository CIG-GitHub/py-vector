from abc import ABC
from abc import abstractmethod

class Backend(ABC):

    # ---- 1. Constructors -------------------------------------------------

    @classmethod
    @abstractmethod
    def from_python(cls, data):
        """
        Create a backend array from a Python iterable.
        Input is always a flat list/tuple coming from PyVector.
        """

    # ---- 2. Introspection ------------------------------------------------

    @abstractmethod
    def length(self) -> int:
        """Return number of elements."""

    @abstractmethod
    def get_item(self, idx: int):
        """Return element at idx (Python scalar)."""

    @abstractmethod
    def to_python(self) -> list:
        """Return storage as plain Python list."""

    # ---- 3. Slicing / Masking -------------------------------------------

    @abstractmethod
    def slice(self, slc: slice):
        """Return a backend representing storage[slc]."""

    @abstractmethod
    def take(self, indices: list[int]):
        """
        Fancy indexing: return a backend with values at given indices.
        Supports boolean mask conversion by PyVector.
        """

    # ---- 4. Elementwise Combination -------------------------------------

    @abstractmethod
    def map_unary(self, func):
        """
        Apply a Python unary function elementwise.
        Returns new backend.
        """

    @abstractmethod
    def map_binary(self, other_backend, func):
        """
        Elementwise combine two backends of equal length.
        func: (x, y) -> z
        Returns new backend.
        """

    # ---- 5. Reductions ---------------------------------------------------

    @abstractmethod
    def reduce(self, func, initial):
        """
        Reduce values with Python reduce-like semantics.
        func: (acc, x) -> acc
        Returns Python scalar.
        """

    # ---- 6. Append / Concatenate ----------------------------------------

    @abstractmethod
    def concat(self, other_backend):
        """
        Return backend representing concatenation:
        self.storage + other.storage
        """
