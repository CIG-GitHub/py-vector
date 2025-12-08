# Representation Rules (repr)

The repr of vectors and tables is designed for human inspection,
not machine round-tripping.

## Vector repr
A vector repr includes:
- first n entries (default 5)
- last n entries (default 5)
- vertical alignment for numeric clarity
- ellipsis if truncated
- footer: `# <length> element <dtype> vector`

Example:
    1
    2
    3
    4
    5
    ...
  995
  996
  997
  998
  999

  # 1000 element vector <int>

## Table repr
A table repr includes:
- column headers
- aligned rows (head/tail)
- ellipsis row separating head/tail
- footer showing shape and dtypes

Column names appear literally, not sanitized.

Example:

col_a   col_b
    1   2025-10-31
    2   2025-10-31
    3   2025-10-31
  ...   ...
  999   2025-10-31
 1000   2025-10-31

# 1000x2 table <int, date>

## Principles
- repr should be unambiguous and legible  
- it should communicate shape, dtype, and sample values  
- it should never attempt to show full large data structures  
- it must remain stable across versions  
- output should resemble notebook head/tail conventions  

