# Vector Rationale

Vector exists to provide a clean, strict, and Python-native foundation for
working with vectors, tables, and multi-dimensional structures without the
complexity, ambiguity, or historical burden of existing dataframe libraries.

The design goals are:

## 1. Semantic clarity
Operations should behave as users intuitively expect from:
- linear algebra (vectorized math, shape preservation)
- tables (column-centric thinking)
- SQL/CSV workflows (row iteration, column selection)
- boolean masking (filtering)

There must be one obvious way to filter, one obvious way to build tables, one
obvious way to index — the library avoids feature duplication or clever magic.

## 2. Column-major conceptual model
Python lists are row-major, but real-world data workflows are column-major:
- CSVs are tall, not wide
- SQL tables are column-defined
- analytics operate column-wise (mean, stdev, mask, sort)

Vector stores tables as a list of vectors, one per column.  
This aligns with actual data usage patterns and produces simpler, cleaner
semantics.

## 3. Predictability over performance
Vector values:
- correctness
- clarity
- obviousness
- exploration speed

…over micro-optimized performance.  
Heavy workloads can migrate to Polars/Arrow; Vector is for the exploratory,
expressive, "human-scale" layer.

## 4. Unambiguous operations
Math preserves shape.  
Boolean masks filter rows.  
Row iteration yields rows.  
Columns never change type.  
Names never mutate due to operations.

This reduces surprises and keeps the mental model coherent.

## 5. Composability into larger systems
Vector is designed to be embedded inside:
- Jupyter notebooks  
- data/analysis workflows  
- interactive canvases  
- computational dashboards  
- input → compute → output pipelines

A consistent vector/table model makes higher layers predictable.

## 6. Minimal API surface
The library prefers:
- one way to slice  
- one way to filter  
- one way to combine columns  
- one table abstraction  
- one vector abstraction  

This keeps the system small, comprehensible, and difficult to misuse.

Vector is intentionally opinionated — features must earn their existence.

