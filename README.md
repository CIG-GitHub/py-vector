# PyVector
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

A clean, typed, composable data layer for Python, built on **PyVector** and **PyTable**.

PyVector provides the foundation; PyTable is your primary tool for readable data modeling and analysis workflows.

## 30-Second Example

```python
from py_vector import PyTable

# Create a table with automatic column name sanitization
t = PyTable({
    "price ($)": [10, 20, 30],
    "quantity":  [4, 5, 6]
})

# Add calculated columns with clean >>= syntax
t >>= (t.price * t.quantity).rename("total")
t >>= (t.total * 0.1).rename("tax")

t
# price ($)    quantity    total    tax
#   [price]  [quantity]  [total]  [tax]
#        10           4       40    4.0
#        20           5      100   10.0
#        30           6      180   18.0
#
# 3×4 table <int, int, int, float>
```

## Installation

```bash
pip install py-vector
```

Zero external dependencies. In a fresh environment:

```bash
pip freeze
# py-vector==0.x.y
```

## Why PyVector?

- Explicit, predictable vector semantics
- Tables compose cleanly from vectors
- Readable "spreadsheet-like" workflows
- Safe defaults (copy-on-write, alias tracking, immutability)
- Immediate visual feedback via `__repr__`
- Zero hidden magic

## Quickstart

### Vectors: elementwise operations

```python
from py_vector import PyVector

a = PyVector([1, 2, 3, 4, 5])
b = PyVector([10, 20, 30, 40, 50])

a + b           # PyVector([11, 22, 33, 44, 55])
a * 2           # PyVector([2, 4, 6, 8, 10])
a > 3           # PyVector([False, False, False, True, True])
```

### Tables: compose vectors with `>>`

```python
from py_vector import PyTable

# Column names auto-sanitize to valid Python attributes
t = PyTable({
    "first name": [1, 2, 3],
    "price ($)":  [10, 20, 30]
})

t.first_name    # PyVector([1, 2, 3])
t.price         # PyVector([10, 20, 30])

# Add columns with >>= (recommended)
t >>= (t.first_name * t.price).rename("total")

t
#   first name  price ($)    total
# [first_name]    [price]  [total]
#            1         10       10
#            2         20       40
#            3         30       90
#
# 3×3 table <int, int, int>
```

### Boolean masking

```python
filtered = t[t.price > 15]

filtered
#   first name  price ($)    total
# [first_name]    [price]  [total]
#            2         20       40
#            3         30       90
#
# 2×3 table <int, int, int>
```

### Joins

```python
customers = PyTable({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
scores = PyTable({'id': [2, 3, 4], 'score': [85, 90, 95]})

result = customers.inner_join(scores, left_on='id', right_on='id')

result
#   id  name            id    score
# [id]  [name]     [id__1]  [score]
#    2  'Bob'            2       85
#    3  'Charlie'        3       90
#
# 2×4 table <int, str, int, int>
```

### Aggregations

```python
t = PyTable({'customer': ['A', 'B', 'A'], 'amount': [100, 200, 150]})

result = t.aggregate(
    over=t.customer,
    sum_over=t.amount,
    count_over=t.amount
)

result
# customer  amount_sum  amount_count
#      'A'         250             2
#      'B'         200             1
#
# 2×3 table <str, int, int>
```

**See [docs/joins-aggregations.md](docs/joins-aggregations.md) for detailed examples.**

## Key Features

### Automatic `__repr__`: Instant Visual Feedback

```python
# Dictionary syntax: quick and familiar
t = PyTable({'id': range(100), 'value': [x**2 for x in range(100)]})

# Or compose from vectors: showcases PyVector's design philosophy
a = PyVector(range(100), name='id')
t = a >> (a**2).rename('value')

t
# id  value
#  0      0
#  1      1
#  2      4
#  3      9
#  4     16
#... ...
# 95   9025
# 96   9216
# 97   9409
# 98   9604
# 99   9801
#
# 100×2 table <int, int>
```

Head/tail preview + type annotations + dimensions—no need for `.head()`, `.info()`, etc.

### Column Name Sanitization

Non-alphanumeric characters become `_`, leading digits get `c` prefix:

```python
t = PyTable({"2023-Q1 Revenue ($M)": [1, 2, 3]})
t.c2023_q1_revenue_m  # Deterministic, predictable access
```

Unnamed columns use system names: `t.col0_`, `t.col1_`, etc.

### Typed Subclasses

PyVector auto-creates typed subclasses with method proxying:

```python
from datetime import date

dates = PyVector([date(2023, 6, 29), date(2024, 1, 2), date(2024, 12, 28)])
dates += 5       # Add 5 days to each date
dates.year       # PyVector([2023, 2024, 2025]) - one crossed the year boundary!
```

Works for `int`, `float`, `str`, `date` types.

## Common Gotchas

### Don't use subscript lists—use boolean masks

```python
# ANTI-PATTERN
indices = [1, 5, 9]
result = v[indices]  # Slow, emits warning

# IDIOMATIC
mask = (v > threshold)
result = v[mask]
```

### Operator overloading: avoid `.index()` on PyVector lists

```python
# WRONG: invokes elementwise equality
cols = [table.year, table.month]
idx = cols.index(table.year)  # Returns boolean vector!

# CORRECT: use enumerate
for idx, col in enumerate(cols):
    if col is table.year:  # identity check
        ...
```

### `None` handling

`None` is excluded from aggregations but counted in `len()`:

```python
v = PyVector([10, None, 20])
sum(v)    # 30 (None excluded)
len(v)    # 3 (None counted)
```

## Design Philosophy

PyVector makes a **strategic choice**: clarity and workflow ergonomics over raw speed.

**What you get:**
- Readable, debuggable code
- No hidden state or aliasing bugs (copy-on-write)
- Deterministic operations
- Zero dependencies
- O(1) fingerprinting for change detection

**When to use PyVector:**
- 10K–1M rows (modeling-scale data)
- Correctness and maintainability matter most
- Jupyter notebook workflows

**When to use alternatives:**
- 10M+ rows → NumPy/Polars
- Deep learning → PyTorch/JAX

## Further Documentation

- **[Performance & Complexity](docs/performance.md)** — O(n) analysis for joins, aggregations, indexing
- **[Exception Handling](docs/exceptions.md)** — Custom exception types and error handling
- **[Aliasing & Fingerprints](docs/aliasing.md)** — Copy-on-write and change detection
- **[Joins & Aggregations](docs/joins-aggregations.md)** — Detailed examples and patterns
- **[Development Guide](docs/development.md)** — Running tests, project structure

## Philosophy

- Clarity beats cleverness
- Explicit beats implicit
- Modeling should feel intuitive
- You should always know what your code is doing

## License
MIT