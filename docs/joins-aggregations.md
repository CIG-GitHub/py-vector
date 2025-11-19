# Joins & Aggregations

## Joins

PyTable supports three join types: `inner_join`, `join` (left join), and `full_join`.

### Inner Join

Returns only rows with matching keys in both tables.

```python
left = PyTable({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
right = PyTable({'id': [2, 3, 4], 'score': [85, 90, 95]})

result = left.inner_join(right, left_on='id', right_on='id')
# Returns rows for id 2 and 3
```

### Left Join

Returns all rows from the left table, with `None` for unmatched right table values.

```python
result = left.join(right, left_on='id', right_on='id')
# Returns all 3 left rows; id=1 has None for score
```

### Full Outer Join

Returns all rows from both tables, with `None` for unmatched values.

```python
result = left.full_join(right, left_on='id', right_on='id')
# Returns 4 rows (1, 2, 3, 4)
```

### Multiple Keys

```python
result = left.inner_join(right, 
    left_on=['year', 'month'], 
    right_on=['year', 'month'])
```

### Cardinality Expectations

Validate join behavior with the `expect` parameter:

```python
result = left.inner_join(right, 
    left_on='id', 
    right_on='customer_id',
    expect='many_to_one')  # Validates right side has unique keys
```

**Complexity:** O(n + m) where n and m are table lengths. Uses hash-based lookups.

---

## Aggregations

### Aggregate: Group and Summarize

Returns **one row per group**.

```python
t = PyTable({
    'customer': ['A', 'B', 'A', 'C', 'B'],
    'amount': [100, 200, 150, 300, 250]
})

result = t.aggregate(
    over=t.customer,
    sum_over=t.amount,
    mean_over=t.amount,
    count_over=t.amount
)
# Returns 3 rows (one per unique customer)
# Columns: customer, amount_sum, amount_mean, amount_count
```

### Window: Running Aggregations

Returns **same row count** as input, with aggregated values repeated per group.

```python
result = t.window(
    over=t.customer,
    sum_over=t.amount
)
# Returns 5 rows (original row count)
# Each row gets the group's total in amount_sum
```

### Multiple Partition Keys

```python
result = t.aggregate(
    over=[t.year, t.month],
    sum_over=t.revenue
)
```

### Custom Aggregation

```python
from functools import reduce
from operator import mul

def prod(vals):
    return reduce(mul, vals, 1)

result = t.aggregate(
    over=t.category,
    apply={'product': (t.value, lambda vals: prod(vals))}
)
```

**Complexity:** O(n) to build partitions, then O(n × k) where k is cost per group aggregation.

---

## Common Patterns

### Join + Aggregate

```python
# Join sales and customers, then aggregate by region
sales = PyTable({'customer_id': [1, 2, 1, 3], 'amount': [100, 200, 150, 300]})
customers = PyTable({'id': [1, 2, 3], 'region': ['East', 'West', 'East']})

joined = sales.join(customers, left_on='customer_id', right_on='id')
result = joined.aggregate(
    over=joined.region,
    sum_over=joined.amount
)
```

### Window for Running Totals

```python
t = PyTable({
    'date': [1, 2, 3, 4, 5],
    'sales': [100, 200, 150, 300, 250]
})

# Cumulative sum (window over entire table)
result = t.window(
    over=PyVector([1] * len(t)),  # Single partition
    sum_over=t.sales
)
```
