# PyVector
A clean, typed, composable data layer for Python.
Built for readable data modeling and analysis workflows.

---

## Installation

    pip install py-vector

PyVector has no external dependencies.
In a fresh environment:

    pip freeze
    # py-vector==0.x.y

This keeps environments clean, predictable, and easy to embed into modeling systems or application code.

---

## Why PyVector exists

PyVector is built for situations where you want:

- explicit, predictable vector semantics
- tables that compose cleanly from vectors
- readable "spreadsheet-like" modeling workflows
- safe defaults (alias tracking, immutable tuples, deterministic operations)
- zero hidden magic

It is intentionally small and easy to reason about.

---

## Quickstart

### Basic vectors

```python
from py_vector import PyVector

a = PyVector([1, 2, 3, 4, 5])
b = PyVector([10, 20, 30, 40, 50])

a + b           # PyVector([11, 22, 33, 44, 55])
a * 2           # PyVector([2, 4, 6, 8, 10])
a > 3           # PyVector([False, False, False, True, True])
```

### Tables from vectors

```python
from py_vector import PyTable

t = PyTable({
    "first name": [1, 2, 3],
    "price ($)":  [10, 20, 30],
    "count@time": [4, 5, 6],
})

# Attributes are sanitized automatically
t.first_name    # PyVector([1, 2, 3])
t.price         # PyVector([10, 20, 30])
t.count_time    # PyVector([4, 5, 6])
```

### Add a new column
Use the `>>` operator to stream new columns into the table. Explicit naming via `.rename()` is recommended.

```python
# Calculate, rename, and append in one step
t = t >> (t.count_time * t.price).rename("total_cost")

# Chain operations cleanly
t = t >> (t.revenue - t.cost).rename("profit") \
      >> (t.profit / t.revenue).rename("margin")
```

### Boolean masking

```python
mask = t.price > 15
filtered = t[mask]
```

### Matrix multiplication

```python
a = PyVector(range(3))

# Create a table by piping vectors
t = a >> a >> a**2

print(t)
# 0  0  0
# 1  1  1
# 2  2  4
#
# 3×3 table <int, int, int>

t.col1_      # Access unnamed column at index 1
t @ t.T      # 3x3 matrix multiplication
```

---

## Core concepts

### PyVector
A typed iterable with value semantics:

- elementwise math and comparison
- optional type safety
- alias tracking prevents accidental mutation of shared data
- fingerprinting for O(1) change detection
- predictable behavior across numeric, string, and date types

### PyTable
A table is a vector of equal-length vectors:

- column-oriented storage
- stable semantics
- deep copy on construction (no aliasing)
- attribute access for valid column names
- fully composable

### Transpose
```python
t.T      # swap rows/columns
```

### Math
Elementwise by default.
Matrix multiplication uses the `@` operator.

---

## Column name sanitization

PyVector converts arbitrary column names into safe Python attributes. It enforces a soft distinction between user data and system internals using trailing underscores.

**User Rules:**
- Non-alphanumeric characters become `_`
- Leading/trailing `_` are **stripped** (e.g., `_price_` becomes `price`)
- Names starting with a digit get a `c` prefix (e.g., `2025` becomes `c2025`)
- Duplicate names get `__1`, `__2` suffixes

**System Rules:**
- Unnamed vectors get reserved system names: `col0_`, `col1_`, etc.
- Because user names always have trailing `_` stripped, `t.col0_` is always safely available for system use.

**Examples:**

```python
t = PyTable({
    "2023-Q1 Revenue ($M)": [1, 2],  # Becomes t.c2023_q1_revenue_m (digit rule)
    "price_": [3, 4],                # Becomes t.price (trailing _ stripped)
    "email": [5, 6],                 # Becomes t.email
})

t.c2023_q1_revenue_m                 # works
t.price                              # works

# Unnamed vectors rely on system names
t = t >> PyVector([7, 8])            # Becomes t.col3_
```

For duplicate names:
```python
t = PyTable({"price ($)": [1, 2], "Price ($)": [3, 4]})
t.price      # first column
t.price__1   # second column
```

---

## Common patterns

### Select multiple columns

```python
subset = t["first name", "count_time"]
```

### Combined boolean logic

```python
mask = (t.price > 10) & (t.count_time < 6)
filtered = t[mask]
```

### Working with dates

```python
from datetime import date

d = PyVector([date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)])
d + 5           # add 5 days
d > date(2024, 1, 2)  # boolean comparison
```

### Nested structures

```python
outer = PyTable([t, t])
outer.size()    # (2, 3, 3)
```

### Ledgers and totals

Append rows using `<<`. Useful for adding totals or creating ledgers.

```python
# Calculate column sums (returns a row vector)
totals = sum(t)

# Append totals to the bottom of the table
t = t << totals
```

---

## Design trade-offs

PyVector prioritizes **clarity and predictable behavior** over raw speed.

What you gain:
- Readable code that's easy to debug
- No hidden state or aliasing bugs
- Deterministic operations
- Zero dependencies

What you trade:
- Not optimized for large-scale numerical computing
- Slower than NumPy/Pandas for multi-million row operations

PyVector performs well for modeling-scale data (thousands to low millions of rows). For workflows where correctness and maintainability matter most, it keeps code clean and explicit.

---

## Philosophy

- Clarity beats cleverness
- Explicit beats implicit
- Modeling should feel intuitive
- Tables are just vectors composed cleanly
- You should always know what your code is doing

---

## License
MIT