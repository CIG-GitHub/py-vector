# PyVector vs Python: Intentional Differences

PyVector is built for analytic workflows, not as a drop-in replacement
for Python lists. In a few places, we *intentionally* depart from Python's
semantics because Python's defaults would produce surprising or unsafe
behavior for data work.

These differences fall into a small, well-defined set:

## 1. No Aliasing (Value Semantics for Vectors)

```python
a = PyVector([1, 2, 3])
b = a
b[1] = 99
```

In Python, this would mutate `a` as well.
PyVector does not allow this.

Each assignment or mutation produces a new vector.
This avoids hidden state and prevents accidental data corruption when
vectors are combined into tables or pipelines.

## 2. Column Behavior Inside Tables

When a vector is inserted into a PyTable, the table receives
a deep copy (value snapshot). Subsequent mutations of the original
vector do not affect the table, and modifying a column in a table
does not affect any external vector.

This ensures table operations are deterministic and isolating.

## 3. Overloaded Operators (`<<` and `>>`)

PyVector overloads two operators that have no standard meaning
for data structures in Python:

- `a >> b` → column-bind two vectors into a PyTable
- `t << v` → append a column to an existing table

These operators are chosen because:

- they express "append/concatenate" visually
- they avoid conflict with Python arithmetic
- they keep table expressions compact in notebooks

If you are accustomed to native Python's meaning of shift operators,
be aware that PyVector uses them exclusively for table assembly.

## 4. Boolean Operators Behave Numerically

Booleans follow Python's numeric rules:

- `True` → `1`
- `False` → `0`

This allows:

```python
(a > 0).sum()   # counts True elements
(b == c) * d    # boolean mask weighted by numeric vector
```

But certain unary operations (e.g., `-b` for boolean `b`) are disabled,
because they have no clear semantic meaning in data analysis.

## 5. Slices, Masks, and Column Indexing Use Data-Model Semantics

PyVector supports a richer indexing model than normal Python:

- column-first indexing
- boolean masks returning filtered vectors
- mixed string + integer column selection in tables
- no aliasing through slicing

These follow analytic conventions (NumPy, R, Pandas, SQL)
rather than Python's built-in list rules.

## Summary

PyVector is Pythonic in syntax but analytic in semantics.

If you are a Python programmer, the two areas to be most aware of are:

1. Vectors use value semantics (no aliasing).
2. Shift operators (`<<` and `>>`) build tables, not bit-shifts.

Everything else behaves naturally once you work with the data model.
