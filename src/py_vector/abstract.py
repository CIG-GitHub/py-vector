# ============================================================
# Abstract Backend + VectorOpsMixin (DECLAUDED)
# ============================================================

from abc import ABC, abstractmethod
import operator


class AbstractVector(ABC):
    """Protocol for all vector-like objects."""

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def schema(self):
        pass


# ============================================================
# VectorOpsMixin (NO LAMBDAS)
# ============================================================

class VectorOpsMixin:
    """
    Mixin defining vector operations.
    Concrete vector must implement:
        __len__
        __getitem__
        _elementwise(op, other)
        _elementwise_reverse(op, other)
        _reduce(fn)
        clone_with(data)
    """

    # --------------------------------------------------------
    # Utility
    # --------------------------------------------------------
    def _binary_prepare(self, other):
        """Return other if lengths match or if it's a scalar."""
        if isinstance(other, AbstractVector):
            if len(self) != len(other):
                raise ValueError("Length mismatch in vector operation")
            return other
        return other

    # --------------------------------------------------------
    # Arithmetic ops (forward)
    # --------------------------------------------------------
    def __add__(self, other):
        return self._elementwise(operator.add, self._binary_prepare(other))

    def __sub__(self, other):
        return self._elementwise(operator.sub, self._binary_prepare(other))

    def __mul__(self, other):
        return self._elementwise(operator.mul, self._binary_prepare(other))

    def __truediv__(self, other):
        return self._elementwise(operator.truediv, self._binary_prepare(other))

    def __floordiv__(self, other):
        return self._elementwise(operator.floordiv, self._binary_prepare(other))

    def __mod__(self, other):
        return self._elementwise(operator.mod, self._binary_prepare(other))

    def __pow__(self, other):
        return self._elementwise(operator.pow, self._binary_prepare(other))

    # --------------------------------------------------------
    # Arithmetic ops (reverse)
    # --------------------------------------------------------
    def __radd__(self, other):
        return self._elementwise_reverse(operator.add, other)

    def __rsub__(self, other):
        return self._elementwise_reverse(operator.sub, other)

    def __rmul__(self, other):
        return self._elementwise_reverse(operator.mul, other)

    def __rtruediv__(self, other):
        return self._elementwise_reverse(operator.truediv, other)

    def __rfloordiv__(self, other):
        return self._elementwise_reverse(operator.floordiv, other)

    def __rmod__(self, other):
        return self._elementwise_reverse(operator.mod, other)

    def __rpow__(self, other):
        return self._elementwise_reverse(operator.pow, other)

    # --------------------------------------------------------
    # Comparisons
    # --------------------------------------------------------
    def __eq__(self, other):
        return self._elementwise(operator.eq, self._binary_prepare(other))

    def __ne__(self, other):
        return self._elementwise(operator.ne, self._binary_prepare(other))

    def __lt__(self, other):
        return self._elementwise(operator.lt, self._binary_prepare(other))

    def __le__(self, other):
        return self._elementwise(operator.le, self._binary_prepare(other))

    def __gt__(self, other):
        return self._elementwise(operator.gt, self._binary_prepare(other))

    def __ge__(self, other):
        return self._elementwise(operator.ge, self._binary_prepare(other))

    # --------------------------------------------------------
    # Reductions
    # --------------------------------------------------------
    def sum(self):
        return self._reduce(operator.add)

    def min(self):
        return self._reduce(min)

    def max(self):
        return self._reduce(max)

    def mean(self):
        return self.sum() / len(self)

    # --------------------------------------------------------
    # Boolean vector truthiness
    # --------------------------------------------------------
    def __bool__(self):
        raise ValueError(
            "The truth value of a PyVector is ambiguous. Use .all() or .any()."
        )

    def all(self):
        return all(self.__iter__())

    def any(self):
        return any(self.__iter__())
