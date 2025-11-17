# Indexing Rules

Indexing in PyVector and PyTable is designed to be strict, predictable, and
easy to reason about. The rules below define exactly what forms of indexing are
allowed and how they behave. The system is intentionally simpler and more
constrained than Pandas or NumPy.

## 1. PyVector Indexing

PyVector follows Python list semantics extended with boolean masking.

### 1.1 Integer index

```python
v[i]
```

Returns a Python scalar at index `i`.

### 1.2 Slice

```python
v[i:j:k]
```

Returns a new PyVector of the same dtype.

### 1.3 Boolean mask

```python
v[mask]
```

Rules:
- `mask` must be a PyVector(bool) of the same length  
- Returns a filtered PyVector  
- Masks may be produced by comparisons, .like(), membership tests, etc.

### 1.4 No advanced indexing

Disallowed:
- lists of indices  
- arrays of indices  
- broadcasting  
- fancy indexing of any form

Only integers, slices, and boolean masks are permitted.

## 2. PyTable Indexing

PyTable supports one-axis indexing (row selection) and two-axis indexing
(row × column). Nothing else is permitted.

The rules below guarantee consistency and eliminate ambiguous cases.

### 2.1 One-axis indexing (row selection)

A single index always refers to rows.

#### Integer

```python
t[i]
```

Returns row `i` as a tuple.

#### Slice

```python
t[i:j]
t[:]
```

Returns a new table containing a subset of rows.  
`t[:]` returns the entire table unchanged.

#### Boolean mask

```python
t[mask]
```

Filters rows across all columns.

Mask rules:
- `mask` must be PyVector(bool)  
- length must match number of rows  
- mask creation is separate from filtering  
- only boolean masks filter rows  

### 2.2 Two-axis indexing: (rows, columns)

Two-axis indexing is allowed only when both axes are integers or slices.
No names, no masks, no lists.

Form:
```python
t[row_index_or_slice, col_index_or_slice]
```

#### Allowed examples

```python
t[1:10, :]
t[:, :]
t[:, 1:3]
t[5:, :]
t[:, 2]
t[10:, 0]
t[5, :]
t[5, 1:4]
t[5, 1]
```

#### Return types

| Form               | Returns      |
|--------------------|--------------|
| `t[i, j]`          | scalar       |
| `t[i, j:k]`        | PyVector     |
| `t[i:j, k]`        | PyVector     |
| `t[i:j, k:l]`      | PyTable      |
| `t[:, :]`          | PyTable      |

### 2.3 Forbidden forms

These forms are disallowed to prevent ambiguity and preserve strict semantics.

#### Masks in second axis

```python
t[:, mask]
t[mask, mask2]
```

Reason: masks filter rows only.

#### Names in second axis

```python
t[:, 'col']
t[mask, 'col']
```

Column names belong to the table API:
```python
t['col']
t.col
```

#### Lists in second axis

```python
t[:, ['a','b']]
```

If multi-column selection is ever added, it will be via:
```python
t[['a','b']]
```
not via two-axis indexing.

## 3. Design Philosophy

- The first index always refers to rows.  
- The second index (when provided) always refers to columns.  
- Boolean masks filter only rows — never columns.  
- Column selection is explicit via names or indices.  
- Slices create rectangular subsets with no surprises.  
- No alignment, no broadcasting, no advanced indexing.  
- `:` is always valid where a slice is valid.  
- `t[:]` returns the full table.  

The goal is a simple, strict, predictable indexing system that cannot be
misinterpreted and does not accumulate the complexity seen in Pandas or NumPy.
