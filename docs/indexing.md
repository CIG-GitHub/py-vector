# Indexing Rules

PyVector and PyTable use a strict, predictable indexing model built around
two principles:

1. **Column-major tables** — columns are primary, rows are derived.
2. **Single-axis indexing only** — but indexing operations are fully composable.

This avoids the complexity of Pandas/NumPy-style multi-axis indexing while
remaining expressive, fast, and mathematically clear.

## 1. PyVector Indexing

A PyVector supports three forms of indexing:

### 1.1 Integer index

```python
v[i]
```

Returns a Python scalar of the underlying dtype.

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
- `mask` must be a PyVector(bool) of the same length.
- Returns a filtered PyVector.
- Masks are produced by comparisons, `.like()`, `.isin()`, etc.

### 1.4 Disallowed forms

- lists of indices  
- arrays of indices  
- broadcasting  
- multi-dimensional indexing  

PyVector deliberately avoids "fancy indexing" to keep semantics simple.

## 2. PyTable Indexing

PyTable supports **only single-axis indexing**, but operations are
composable: column selection followed by row slicing (or vice-versa).

This preserves clarity and matches the column-major design.

### 2.1 Row Indexing

A single index always refers to **rows**.

#### Integer

```python
t[i]
```

Returns row `i` as a tuple-like record.

#### Slice

```python
t[i:j]
t[:]
```

Returns a new PyTable containing a subset of rows.

#### Boolean mask

```python
t[mask]
```

Filters rows across all columns.

Mask rules:
- must be a PyVector(bool) of length equal to the number of rows
- mask creation is separate from filtering
- **only boolean masks filter rows** (never columns)

### 2.2 Column Selection

Columns may be selected via:

#### String name

```python
t['col']
t.col
```

Returns a PyVector.

#### Tuple of names (multi-column select)

```python
t['a', 'b', 'c']
```

Returns a new PyTable containing those columns (in order).

**Name resolution:**
- Full column names are always valid.
- Disambiguated names are also valid (first occurrence of a duplicate name).
- Python passes this as a single tuple key — no conflict with row indexing.

#### Column index selection

Using the helper:

```python
t.cols([2, 5, 7])
t.cols(slice(3, 8))
```

Returns a new PyTable with only the specified columns.

#### Chaining is fully supported

```python
t['a', 'b'][10:20]
t[mask]['x', 'y']
t.cols([1, 3, 4])[5:]
```

This replaces all two-axis forms such as `t[rows, cols]`.

### 2.3 Forbidden Forms

The following are intentionally disallowed to avoid ambiguity:

#### No two-axis indexing

```python
t[i, j]
t[i, j:k]
t[i:j, k:l]
```

PyTable never interprets a tuple of length 2 as multi-axis indexing.

#### No list × list indexing

```python
t[[1, 2], [3, 4]]
```

Avoids Pandas/NumPy complex rules and shape ambiguities.

#### No masks on columns

```python
t[:, mask]
t[mask, 'col']
```

Masks apply **only** to the row axis.

#### No name-based column selection inside two-axis forms

Since two-axis indexing is removed entirely, the following are invalid:

```python
t[:, 'a']
t[5, 'b']
```

Column names are always used in **single-axis column selection**.

## 3. Recommended Idioms

### Column-first selection (preferred for large datasets)

Because PyTable is column-major, selecting columns first reduces the memory
footprint and speeds up operations:

```python
t['a', 'b'][1000:2000]
t.cols([2, 4, 5])[mask]
```

### Row-first selection (equally valid)

```python
t[1000:2000]['a', 'b']
```

Composability guarantees these two forms are equivalent.

### Mask → column select

```python
t[mask]['col']
t['col'][mask]
```

Either direction is legal and predictable.

## 4. Design Philosophy

- PyTable is column-major; columns are the primary structural axis.
- Indexing is single-axis, but fully composable.
- Only integer, slice, or boolean mask indexing is allowed on rows.
- Only names or `.cols()` are allowed for column selection.
- Multi-column selection via `t['a', 'b', 'c']` is first-class and preferred.
- Both full names and disambiguated names are valid for column selection.
- No advanced indexing, no broadcasting, no silent reindexing.
- If an operation is legal in one context, it is legal everywhere else.

The model is intentionally strict, minimal, and easy to reason about.
It is designed for clarity, correctness, and high user ergonomics rather than
feature maximalism.
