# Vector Semantics

A Vector represents:
- a one-dimensional, fixed-length, single-dtype sequence  
- supporting elementwise operations  
- with consistent indexing, slicing, masking, and repr behavior  

## 1. Shape
Vectors are conceptually **column vectors by default**, rendering vertically.
Horizontal vectors appear only when the user explicitly requests `.T`.

## 2. Indexing
`v[i]` returns a scalar.  
`slices` return new vectors with preserved dtype.

## 3. Math
Binary math:
- requires same length  
- is elementwise  
- produces a new vector with the same dtype (except standard numeric coercions)

Unary math preserves dtype.

## 4. Boolean masks
A boolean vector `mask` must match length.  
Filtering: `v[mask]` produces a shorter vector.

## 5. Iteration
`for x in v:` yields scalars, increasing index order.

## 6. Copying
`v.copy()` reproduces:
- values  
- dtype  
- name  

Most other operations erase name unless explicitly copying.

## 7. Repr
Vectors show:
- first 5  
- last 5  
- aligned digits  
- numeric clarity  
- shape + dtype footer

Vertical display emphasizes table-first design.

