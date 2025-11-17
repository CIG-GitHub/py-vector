# Naming Semantics

Names in PyVector and PyTable are simple, non-magical metadata.

## PyVector
- `name` is optional  
- math operations do not propagate names  
- slicing does not preserve name  
- copying preserves name  

Names are human-facing, not structural.

## PyTable
- each column has its own name  
- duplicate names allowed  
- dot-access resolves only to the first match  
- attribute names are sanitized versions of original names  
- sanitization rules:
  - only [A-Za-z0-9_]
  - collapse consecutive `_`
  - strip leading `_`
  - if collision: add `__2`, `__3`, …

Sanitized names do not replace the actual names.

## Practical consequences
- weird user-provided column names are allowed  
- dot-access provides ergonomic access without polluting data semantics  
- users may explicitly rename columns if needed  
