import warnings
from typing import Any

# Adjust imports to point to your new structure
from .vector import PyVector
from .naming import _sanitize_user_name
from .naming import _uniquify
from .errors import PyVectorKeyError
from .errors import PyVectorValueError
from .errors import PyVectorTypeError
from .backends.table import TableBackend

def _missing_col_error(name, context="PyTable"):
    return PyVectorKeyError(f"Column '{name}' not found in {context}")


class _RowView:
    """Lightweight row view for iterating over table rows with attribute access."""
    __slots__ = ('_cols_data', '_column_map', '_index')
    
    def __init__(self, table, index):
        # Cache direct handles to underlying raw tuple data
        self._cols_data = [col._impl._data for col in table._impl._data]

        # Build name -> column_index map from PyTable's column map
        cmap = table._build_column_map()
        # cmap: name -> (col, idx)
        self._column_map = {name: idx for name, (_col, idx) in cmap.items()}

        self._index = index

    def set_index(self, index):
        self._index = index
        return self

    def __getattr__(self, attr):
        col_idx = self._col_map.get(attr)
        if col_idx is None:
            raise AttributeError(f"Row has no attribute '{attr}'")
        return self._cols_data[col_idx][self._index]

    def __getitem__(self, key):
        # row[0] / row[1] etc.
        if isinstance(key, int):
            return self._cols_data[key][self._index]
        # row["a"]
        if isinstance(key, str):
            return getattr(self, key)
        raise TypeError(f"Row indices must be int or str, not {type(key).__name__}")

    def __iter__(self):
        idx = self._index
        for col_data in self._cols_data:
            yield col_data[idx]

    def __len__(self):
        return len(self._cols_data)

    def __repr__(self):
        # Only used in tests / debugging; keep it cheap
        try:
            from .display import _printr
            return _printr(self)
        except ImportError:
            return f"<RowView index={self._index} cols={len(self._cols_data)}>"



class PyTable(PyVector):

    def __new__(cls, initial=(), dtype=None, name=None, as_row=False, _impl=None):
        """
        PyTable constructor (factory).
        If _impl is provided, wrap it directly (same pattern as PyVector).
        Otherwise, build a TableBackend from the given columns/rows.
        """
        if _impl is not None:
            # Direct backend wrap (coming from joins, slicing, etc.)
            obj = super().__new__(cls)
            super(PyTable, obj).__init__(None, dtype=None, name=name, _impl=_impl)
            return obj

        # Normalize input
        # Case: dict → columns with names
        if isinstance(initial, dict):
            cols = [PyVector(values, name=key) for key, values in initial.items()]
        else:
            # Case: sequence of columns or rows
            cols = []
            for item in initial:
                if isinstance(item, PyVector):
                    cols.append(item)
                else:
                    cols.append(PyVector(item))

        # Validate equal length
        nrows = len(cols[0]) if cols else 0
        for col in cols:
            if len(col) != nrows:
                warnings.warn("Creating PyTable with columns of unequal length.", UserWarning)

        # Build backend from raw data
        raw_cols  = tuple(col._impl._data for col in cols)
        col_names = tuple(col.name for col in cols)

        backend = TableBackend(raw_cols, names=col_names, nrows=nrows, name=name)

        # Create shell object with backend
        obj = super().__new__(cls)
        super(PyTable, obj).__init__(None, dtype=None, name=name, _impl=backend)
        return obj


    def __init__(self, *args, **kwargs):
        """
        PyTable.__init__ does nothing.

        Construction logic is entirely done in __new__,
        mirroring ndarray, pandas, and our PyVector(_impl=...) pattern.
        """
        pass


    def __len__(self):
        if not self._impl._data:
            return 0
        # If the first element is a Table (nested), recurse? 
        # For standard tables, return row count.
        # But wait - PyVector.__len__ returns len(self._impl).
        # self._impl is TupleBackend([col1, col2]). len is num_cols.
        # Tables usually want len() to be num_rows.
        
        # If we follow Pandas semantics: len(df) is num_rows.
        return self._length

    def size(self):
        # Rows, Cols
        return (self._length, len(self._impl))

    @property
    def cols(self):
        """
        Returns a simple dict:
            name -> PyVector
        """
        cmap = self._build_column_map()
        return {name: col for name, (col, _) in cmap.items()}

    def _build_column_map(self):
        """
        Returns:
            dict mapping:
                sanitized_name -> (column PyVector, original_index)

        Guarantees:
            - Sanitization is stable
            - Duplicate sanitized names get suffixes: col, col_1, col_2, ...
            - Order is preserved
        """
        from .naming import _sanitize_user_name

        result: dict[str, tuple[PyVector, int]] = {}
        used: set[str] = set()

        # only use backend data here
        cols = self._impl._data  # tuple of column PyVectors

        for idx, col in enumerate(cols):
            raw = col.name  # public shell property
            base = _sanitize_user_name(raw if raw else f"col{idx}")

            # Ensure unique sanitized name
            name = base
            counter = 1
            while name in used:
                name = f"{base}_{counter}"
                counter += 1

            used.add(name)
            result[name] = (col, idx)

        return result


    def __dir__(self):
        from .naming import _sanitize_user_name
        base_attrs = set(super().__dir__())
        
        # Generate sanitized names for all current columns
        col_attrs = set()
        for col in self._impl._data:
            s = _sanitize_user_name(col.name)
            if s:
                col_attrs.add(s)
                
        return list(col_attrs | base_attrs)


    def __getattr__(self, attr):
        """
        Allows t.customer, t.amount, etc.
        Required by tests.
        """
        cmap = self._build_column_map()

        if attr in cmap:
            return cmap[attr][0]

        raise AttributeError(f"'PyTable' object has no attribute '{attr}'")
    
    def rename_column(self, old_name, new_name):
        # Just find the column and mutate it. No map update needed.
        for col in self._impl._data:
            if col.name == old_name:
                col.rename(new_name) # In-place mutation of the column shell
                return self
        raise _missing_col_error(old_name)
    
    def rename_columns(self, old_names, new_names):
        if len(old_names) != len(new_names):
            raise PyVectorValueError("old_names and new_names must have the same length")

        # 1. Resolve Targets (Atomic Check)
        # We need to map each (old, new) request to a specific Column Object.
        targets = [] # List of (ColumnObject, NewName)
        
        # Track which columns we've already "claimed" for renaming in this batch
        # so we don't pick the same 'a' twice for two different rules.
        claimed_indices = set() 
        
        for old, new in zip(old_names, new_names):
            found = False
            for idx, col in enumerate(self._impl._data):
                if idx in claimed_indices:
                    continue
                
                if col.name == old:
                    targets.append((col, new))
                    claimed_indices.add(idx)
                    found = True
                    break
            
            if not found:
                # Atomic failure: If one missing, fail all BEFORE renaming anything
                raise _missing_col_error(old)

        # 2. Apply Renames
        for col, new_name in targets:
            col.rename(new_name)
            
        return self

    @property
    def T(self):
        if len(self.size()) == 2:
            num_rows = self._length
            rows = []
            columns = self._impl._data
            # Transpose: each row becomes a Vector
            for row_idx in range(num_rows):
                # Optimize: grab raw data from cols
                row_vals = [c._impl._data[row_idx] for c in columns]
                rows.append(PyVector(row_vals))
            return PyTable(rows)
        return self.copy((tuple(x.T for x in self.cols())))

    def __getitem__(self, key):
        # Column selection: t['a']
        if isinstance(key, str):
            # exact name lookup
            for col in self._impl._data:
                if col.name == key:
                    return col
            
            # Try sanitized name lookup
            key_lower = key.lower()
            seen = set()
            for idx, col in enumerate(self._impl._data):
                if col.name is not None:
                    base = _sanitize_user_name(col.name)
                    if base is None:
                        if f'col{idx}_' == key_lower:
                            return col
                    else:
                        sanitized = _uniquify(base, seen)
                        seen.add(sanitized)
                        if sanitized == key_lower:
                            return col
                else:
                    if f'col{idx}_' == key_lower:
                        return col
            
            raise _missing_col_error(key)

        # Multiple columns: t['a','b']
        if isinstance(key, tuple):
            selected = [self[k] for k in key]
            return PyTable(selected)

        # Row indexing: t[3], t[2:5], etc.
        if isinstance(key, int):
            # return a row as a list of scalar values
            return [col[key] for col in self._impl._data]

        if isinstance(key, slice):
            # return table of sliced columns
            sliced = [col[key] for col in self._impl._data]
            return PyTable(sliced)

        if isinstance(key, PyVector):
            # boolean mask or integer gather
            seq = list(key)
            sliced = [col[seq] for col in self._impl._data]
            return PyTable(sliced)

        raise TypeError(f"Invalid index type for PyTable: {type(key)}")


    def __iter__(self):
        """
        Iterate over rows.

        Yields _RowView objects with attribute access:
            for row in t:
                row.customer
                row["amount"]
        """
        from .naming import _sanitize_user_name

        backend = self._impl
        cols_data = backend._data
        nrows = self._length

        # Build attribute → column index map once
        attr_map = {}
        used = set()

        # We get raw names from backend._names, which was passed in at construction.
        names = getattr(backend, "_names", tuple(None for _ in cols_data))

        for idx, raw_name in enumerate(names):
            base = _sanitize_user_name(raw_name if raw_name is not None else f"col{idx}")
            if not base:
                continue

            name = base
            counter = 1
            while name in used:
                name = f"{base}_{counter}"
                counter += 1

            used.add(name)
            attr_map[name] = idx

        # Reuse a single RowView, just changing its index
        row_view = _RowView(cols_data, attr_map, 0)

        for i in range(nrows):
            row_view.set_index(i)
            yield row_view


    def __repr__(self):
        # assuming _printr handles PyTable logic
        try:
            from .display import _printr
            return _printr(self)
        except ImportError:
            return f"PyTable({self.size()} rows)"

    def __matmul__(self, other):
        if isinstance(other, PyVector):
            if other.ndims() == 1:
                return PyVector(tuple(sum(u*v for u, v in zip(self._impl._data, other._impl._data)) for s in self))
        return super().__matmul__(other)

    def _elementwise_compare(self, other, op):
        # Implementation depends on desired semantics for Table == Table
        # For now, simplistic vector of bools?
        pass

    def __rshift__(self, other):
        # Adding columns
        cols = list(self._impl._data)
        
        if isinstance(other, PyTable):
            cols.extend(other._impl._data)
        elif isinstance(other, PyVector):
            cols.append(other)
        elif hasattr(other, '__iter__') and not isinstance(other, (str, bytes)):
             cols.append(PyVector(other))
        else:
             cols.append(PyVector((other,)))
             
        return PyTable(cols)

    def __lshift__(self, other):
        # Concatenate rows (append)
        if isinstance(other, PyTable):
            new_cols = [x << y for x, y in zip(self._impl._data, other._impl._data)]
            return PyTable(new_cols)
        # Handle dict append?
        return super().__lshift__(other)

    # -------------------------------------------------------------------------
    # Joins - Updated to use ._impl._data for raw access
    # -------------------------------------------------------------------------

    def _validate_join_keys(self, other, left_on, right_on, expect=None):
        """
        Normalize and validate join key specifications.

        left_on, right_on:
          - str  -> column name (supports sanitized lookup)
          - PyVector -> explicit key column (must match table length)
          - list/tuple of the above

        Raises
        ------
        ValueError
            If left_on/right_on lists have different lengths.
        PyVectorValueError
            If key column lengths don't match their tables, or each other.
        PyVectorTypeError
            If key dtypes are incompatible.
        """
        from collections import Counter
        from .vector import PyVector  # local import to avoid cycles

        # Normalize to lists
        if isinstance(left_on, (str, PyVector)):
            left_on = [left_on]
        if isinstance(right_on, (str, PyVector)):
            right_on = [right_on]

        if len(left_on) != len(right_on):
            # This is a *specification* mismatch, not data -> plain ValueError
            raise ValueError("left_on and right_on must have the same length")

        def get_column(table, col_spec, side_name):
            # Strings: delegate to table.__getitem__ which already does
            # exact + sanitized lookup and raises PyVectorKeyError
            if isinstance(col_spec, str):
                try:
                    return table[col_spec]
                except Exception as e:
                    # Preserve PyVectorKeyError, rewrap everything else
                    from .errors import PyVectorKeyError
                    if isinstance(e, PyVectorKeyError):
                        raise
                    raise PyVectorKeyError(
                        f"Column '{col_spec}' not found in {side_name} table"
                    )

            # Explicit PyVector key column
            if isinstance(col_spec, PyVector):
                return col_spec

            # Anything else is an invalid type for key spec
            from .errors import PyVectorValueError
            raise PyVectorValueError(f"Invalid column spec: {type(col_spec)}")

        from .errors import PyVectorValueError, PyVectorTypeError

        left_cols: list[PyVector] = []
        right_cols: list[PyVector] = []

        for i, (l_spec, r_spec) in enumerate(zip(left_on, right_on)):
            l_col = get_column(self, l_spec, "left")
            r_col = get_column(other, r_spec, "right")

            # Length checks
            if len(l_col) != len(self):
                raise PyVectorValueError(
                    f"Join key length mismatch on left (position {i})"
                )
            if len(r_col) != len(other):
                raise PyVectorValueError(
                    f"Join key length mismatch on right (position {i})"
                )
            if len(l_col) != len(r_col):
                raise PyVectorValueError(
                    f"Join key columns must have same length at position {i}"
                )

            # DType / kind compatibility
            ls = l_col.schema()
            rs = r_col.schema()
            if ls and rs and ls.kind != rs.kind:
                raise PyVectorTypeError(
                    f"Join key mismatch at position {i}: {ls.kind} vs {rs.kind}"
                )

            left_cols.append(l_col)
            right_cols.append(r_col)

        # Cardinality checks (optional)
        # expect ∈ {"many_to_many","many_to_one","one_to_many","one_to_one"} or None
        if expect and expect != "many_to_many":
            # Composite keys: tuples across all key columns
            if left_cols:
                left_keys = list(zip(*[c._impl._data for c in left_cols]))
                right_keys = list(zip(*[c._impl._data for c in right_cols]))
            else:
                left_keys = []
                right_keys = []

            lc = Counter(left_keys)
            rc = Counter(right_keys)

            # one_to_* => left side must not repeat keys
            if expect in ("one_to_one", "one_to_many"):
                if any(cnt > 1 for cnt in lc.values()):
                    raise ValueError(
                        "Join cardinality violation: left side is not one-to-*"
                    )

            # *_to_one => right side must not repeat keys
            if expect in ("one_to_one", "many_to_one"):
                if any(cnt > 1 for cnt in rc.values()):
                    raise ValueError(
                        "Join cardinality violation: right side is not *-to-one"
                    )

        # Return list[(left_col, right_col)]
        return list(zip(left_cols, right_cols))


    @staticmethod
    def _validate_key_tuple_hashable(key_tuple, key_cols, row_idx):
        try:
            hash(key_tuple)
        except TypeError as e:
            raise PyVectorTypeError(f"Unhashable key at row {row_idx}: {e}")


    def join(self, other, left_on, right_on, expect='many_to_one'):
        """
        High-performance join with cardinality expectations.
        """

        # -------------------------
        # 1. Validate expectation
        # -------------------------
        if expect not in ('one_to_one', 'many_to_one', 'one_to_many', 'many_to_many'):
            raise PyVectorValueError(
                f"Invalid expect '{expect}'. "
                "Must be one of 'one_to_one', 'many_to_one', 'one_to_many', 'many_to_many'."
            )

        check_right_unique = expect in ('one_to_one', 'many_to_one')
        check_left_unique  = expect in ('one_to_one', 'one_to_many')

        # -------------------------
        # 2. Resolve key columns
        # -------------------------
        pairs = self._validate_join_keys(other, left_on, right_on)

        left_keys_data  = [col._impl._data for col, _ in pairs]
        right_keys_data = [col._impl._data for _, col in pairs]

        pk_len = len(left_keys_data)

        # -------------------------
        # 3. Access underlying data
        # -------------------------
        left_cols  = self._impl._data
        right_cols = other._impl._data

        left_cols_data  = [c._impl._data for c in left_cols]
        right_cols_data = [c._impl._data for c in right_cols]

        left_nrows  = len(self)
        right_nrows = len(other)

        # -------------------------
        # 4. Build hash index on RIGHT
        # -------------------------
        right_index = {}
        if check_right_unique:
            duplicates = {}

        for r_idx in range(right_nrows):
            key = tuple(right_keys_data[k][r_idx] for k in range(pk_len))

            bucket = right_index.get(key)
            if bucket is None:
                right_index[key] = [r_idx]
            else:
                bucket.append(r_idx)
                if check_right_unique and key not in duplicates:
                    duplicates[key] = bucket

        # Enforce right-side uniqueness
        if check_right_unique and duplicates:
            key, occurrences = next(iter(duplicates.items()))
            raise PyVectorValueError(
                f"Join expectation '{expect}' violated: right side key {key} "
                f"appears {len(occurrences)} times."
            )

        # -------------------------
        # 5. Track left-side uniqueness if needed
        # -------------------------
        if check_left_unique:
            left_seen = set()

        # -------------------------
        # 6. Allocate result columns
        # -------------------------
        total_cols = len(left_cols_data) + len(right_cols_data)
        result = [[] for _ in range(total_cols)]

        # -------------------------
        # 7. Join loop (left-outer)
        # -------------------------
        for l_idx in range(left_nrows):
            key = tuple(left_keys_data[k][l_idx] for k in range(pk_len))

            if check_left_unique:
                if key in left_seen:
                    raise PyVectorValueError(
                        f"Join expectation '{expect}' violated: left side key {key} "
                        f"appears multiple times."
                    )
                left_seen.add(key)

            matches = right_index.get(key)

            if matches:
                for r_idx in matches:
                    # left side
                    for i, coldata in enumerate(left_cols_data):
                        result[i].append(coldata[l_idx])
                    # right side
                    base = len(left_cols_data)
                    for j, coldata in enumerate(right_cols_data):
                        result[base + j].append(coldata[r_idx])
            else:
                # left row + None placeholders
                for i, coldata in enumerate(left_cols_data):
                    result[i].append(coldata[l_idx])
                base = len(left_cols_data)
                for j in range(len(right_cols_data)):
                    result[base + j].append(None)

        # -------------------------
        # 8. Wrap into PyVectors
        # -------------------------
        res_cols = []
        idx = 0

        # left columns
        for col in left_cols:
            res_cols.append(PyVector(result[idx], name=col.name))
            idx += 1

        # right columns
        for col in right_cols:
            res_cols.append(PyVector(result[idx], name=col.name))
            idx += 1

        return PyTable(res_cols)


    def inner_join(self, other, left_on, right_on, expect='many_to_one'):
        # -------------------------
        # 1. Validate cardinality expectation
        # -------------------------
        if expect not in ('one_to_one', 'many_to_one', 'one_to_many', 'many_to_many'):
            raise PyVectorValueError(
                f"Invalid expect='{expect}'. Must be one of "
                "'one_to_one', 'many_to_one', 'one_to_many', 'many_to_many'."
            )

        check_right_unique = expect in ('one_to_one', 'many_to_one')
        check_left_unique  = expect in ('one_to_one', 'one_to_many')

        # -------------------------
        # 2. Resolve join key columns
        # -------------------------
        pairs = self._validate_join_keys(other, left_on, right_on)

        # Raw data access for speed (your new pattern)
        left_keys_data  = [col._impl._data for col, _ in pairs]
        right_keys_data = [col._impl._data for _, col in pairs]

        pk_len = len(left_keys_data)

        # Underlying data columns
        left_cols      = self._impl._data
        right_cols     = other._impl._data
        left_cols_data  = [c._impl._data for c in left_cols]
        right_cols_data = [c._impl._data for c in right_cols]

        left_nrows  = len(self)
        right_nrows = len(other)

        # -------------------------
        # 3. Build RIGHT hash index
        # -------------------------
        right_index = {}
        if check_right_unique:
            duplicates = {}

        for r_idx in range(right_nrows):
            key = tuple(right_keys_data[k][r_idx] for k in range(pk_len))
            bucket = right_index.get(key)
            if bucket is None:
                right_index[key] = [r_idx]
            else:
                bucket.append(r_idx)
                if check_right_unique:
                    duplicates[key] = bucket

        # Enforce right uniqueness
        if check_right_unique and duplicates:
            key, occ = next(iter(duplicates.items()))
            raise PyVectorValueError(
                f"Join expectation '{expect}' violated: right key {key} "
                f"appears {len(occ)} times."
            )

        # -------------------------
        # 4. Track LEFT uniqueness if needed
        # -------------------------
        if check_left_unique:
            left_seen = set()

        # -------------------------
        # 5. Allocate result columns
        # -------------------------
        total_cols = len(left_cols_data) + len(right_cols_data)
        result = [[] for _ in range(total_cols)]

        # -------------------------
        # 6. Perform INNER JOIN
        # -------------------------
        for l_idx in range(left_nrows):
            key = tuple(left_keys_data[k][l_idx] for k in range(pk_len))

            if check_left_unique:
                if key in left_seen:
                    raise PyVectorValueError(
                        f"Join expectation '{expect}' violated: left key {key} "
                        f"appears more than once."
                    )
                left_seen.add(key)

            matches = right_index.get(key)
            if not matches:
                continue  # inner join → skip unmatched

            for r_idx in matches:
                # left side
                for i, coldata in enumerate(left_cols_data):
                    result[i].append(coldata[l_idx])

                # right side
                base = len(left_cols_data)
                for j, coldata in enumerate(right_cols_data):
                    result[base + j].append(coldata[r_idx])

        # -------------------------
        # 7. Empty result → empty table
        # -------------------------
        if all(len(col) == 0 for col in result):
            return PyTable([])

        # -------------------------
        # 8. Wrap into PyVectors
        # -------------------------
        res_cols = []
        idx = 0

        # left columns
        for col in left_cols:
            res_cols.append(PyVector(result[idx], name=col.name))
            idx += 1

        # right columns
        for col in right_cols:
            res_cols.append(PyVector(result[idx], name=col.name))
            idx += 1

        return PyTable(res_cols)


    def full_join(self, other, left_on, right_on, expect='many_to_many'):
        # -------------------------
        # 1. Validate expectation
        # -------------------------
        if expect not in ('one_to_one', 'many_to_one', 'one_to_many', 'many_to_many'):
            raise PyVectorValueError(
                f"Invalid expect='{expect}'. Must be one of "
                "'one_to_one', 'many_to_one', 'one_to_many', 'many_to_many'."
            )

        check_right_unique = expect in ('one_to_one', 'many_to_one')
        check_left_unique  = expect in ('one_to_one', 'one_to_many')

        # -------------------------
        # 2. Resolve join keys
        # -------------------------
        pairs = self._validate_join_keys(other, left_on, right_on)

        left_keys_data  = [col._impl._data for col, _ in pairs]
        right_keys_data = [col._impl._data for _, col in pairs]
        pk_len = len(left_keys_data)

        # Raw access to columns (fast-path)
        left_cols  = self._impl._data
        right_cols = other._impl._data

        left_cols_data  = [c._impl._data for c in left_cols]
        right_cols_data = [c._impl._data for c in right_cols]

        left_nrows  = len(self)
        right_nrows = len(other)

        # -------------------------
        # 3. Build RIGHT index
        # -------------------------
        right_index = {}
        if check_right_unique:
            duplicates = {}

        for r_idx in range(right_nrows):
            key = tuple(right_keys_data[k][r_idx] for k in range(pk_len))

            bucket = right_index.get(key)
            if bucket is None:
                right_index[key] = [r_idx]
            else:
                bucket.append(r_idx)
                if check_right_unique:
                    duplicates[key] = bucket

        # Enforce right-side uniqueness
        if check_right_unique and duplicates:
            key, occ = next(iter(duplicates.items()))
            raise PyVectorValueError(
                f"Join expectation '{expect}' violated: right key {key} "
                f"appears {len(occ)} times."
            )

        # -------------------------
        # 4. Left-side uniqueness + track right matches
        # -------------------------
        if check_left_unique:
            left_seen = set()

        matched_right = set()
        matched_right_add = matched_right.add

        # -------------------------
        # 5. Allocate result columns
        # -------------------------
        total_cols = len(left_cols_data) + len(right_cols_data)
        result = [[] for _ in range(total_cols)]

        # -------------------------
        # 6. Process LEFT table (outer join left side)
        # -------------------------
        for l_idx in range(left_nrows):
            key = tuple(left_keys_data[k][l_idx] for k in range(pk_len))

            # enforce left cardinality rule
            if check_left_unique:
                if key in left_seen:
                    raise PyVectorValueError(
                        f"Join expectation '{expect}' violated: left key {key} appears more than once."
                    )
                left_seen.add(key)

            matches = right_index.get(key)

            if matches:
                for r_idx in matches:
                    matched_right_add(r_idx)

                    # left columns
                    for i, coldata in enumerate(left_cols_data):
                        result[i].append(coldata[l_idx])

                    # right columns
                    base = len(left_cols_data)
                    for j, coldata in enumerate(right_cols_data):
                        result[base + j].append(coldata[r_idx])
            else:
                # no match on right → None placeholders
                for i, coldata in enumerate(left_cols_data):
                    result[i].append(coldata[l_idx])
                base = len(left_cols_data)
                for j in range(len(right_cols_data)):
                    result[base + j].append(None)

        # -------------------------
        # 7. Append unmatched RIGHT rows
        # -------------------------
        for r_idx in range(right_nrows):
            if r_idx not in matched_right:
                # left side = None
                for i in range(len(left_cols_data)):
                    result[i].append(None)

                # right side = actual values
                base = len(left_cols_data)
                for j, coldata in enumerate(right_cols_data):
                    result[base + j].append(coldata[r_idx])

        # -------------------------
        # 8. Wrap into PyVectors
        # -------------------------
        res_cols = []
        idx = 0

        # left
        for col in left_cols:
            res_cols.append(PyVector(result[idx], name=col.name))
            idx += 1

        # right
        for col in right_cols:
            res_cols.append(PyVector(result[idx], name=col.name))
            idx += 1

        return PyTable(res_cols)


    def aggregate(
        self,
        over,
        sum_over=None,
        mean_over=None,
        min_over=None,
        max_over=None,
        count_over=None,
        stdev_over=None,     # NEW
        apply=None,
    ):
        from .errors import PyVectorValueError
        import math

        # ------------------------------------------------------------------
        # 1. Normalize partition keys
        # ------------------------------------------------------------------
        if isinstance(over, PyVector):
            over = [over]

        nrows = self._length

        # Partition key length check
        for k in over:
            if len(k) != nrows:
                # required by tests
                raise ValueError("Partition key length mismatch")

        # Convert single-or-list args → list
        def _ensure_list(x):
            if x is None:
                return []
            return x if isinstance(x, list) else [x]

        sum_cols   = _ensure_list(sum_over)
        mean_cols  = _ensure_list(mean_over)
        min_cols   = _ensure_list(min_over)
        max_cols   = _ensure_list(max_over)
        count_cols = _ensure_list(count_over)
        stdev_cols = _ensure_list(stdev_over)

        # Collect all aggregation columns for length checking
        agg_cols = sum_cols + mean_cols + min_cols + max_cols + count_cols + stdev_cols

        if apply:
            for col, _func in apply.values():
                agg_cols.append(col)

        # Aggregation column length checks
        for col in agg_cols:
            if len(col) != nrows:
                raise ValueError("Aggregation column length mismatch")

        # ------------------------------------------------------------------
        # 2. Build groups: key -> list of row indices
        # ------------------------------------------------------------------
        over_data = [c._impl._data for c in over]
        pk_len = len(over)

        groups = {}
        for idx in range(nrows):
            key = tuple(over_data[k][idx] for k in range(pk_len))
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)

        group_keys = list(groups.keys())  # stable order for deterministic output
        result_cols = []

        # ------------------------------------------------------------------
        # 3. Reconstruct partition-key columns
        # ------------------------------------------------------------------
        for k_idx, k_col in enumerate(over):
            vals = [gk[k_idx] for gk in group_keys]
            result_cols.append(PyVector(vals, name=k_col.name))

        # ------------------------------------------------------------------
        # 4. Aggregations
        # ------------------------------------------------------------------

        # SUM
        for col in sum_cols:
            raw = col._impl._data
            out = []
            for gk in group_keys:
                total = 0
                for i in groups[gk]:
                    v = raw[i]
                    if v is not None:
                        total += v
                out.append(total)
            name = f"{col.name}_sum" if col.name else "sum"
            result_cols.append(PyVector(out, name=name))

        # COUNT
        for col in count_cols:
            raw = col._impl._data
            out = []
            for gk in group_keys:
                cnt = 0
                for i in groups[gk]:
                    if raw[i] is not None:
                        cnt += 1
                out.append(cnt)
            name = f"{col.name}_count" if col.name else "count"
            result_cols.append(PyVector(out, name=name))

        # MEAN
        for col in mean_cols:
            raw = col._impl._data
            out = []
            for gk in group_keys:
                total = 0
                cnt = 0
                for i in groups[gk]:
                    v = raw[i]
                    if v is not None:
                        total += v
                        cnt += 1
                out.append(total / cnt if cnt else None)
            name = f"{col.name}_mean" if col.name else "mean"
            result_cols.append(PyVector(out, name=name))

        # MIN
        for col in min_cols:
            raw = col._impl._data
            out = []
            for gk in group_keys:
                curr_min = None
                for i in groups[gk]:
                    v = raw[i]
                    if v is not None and (curr_min is None or v < curr_min):
                        curr_min = v
                out.append(curr_min)
            name = f"{col.name}_min" if col.name else "min"
            result_cols.append(PyVector(out, name=name))

        # MAX
        for col in max_cols:
            raw = col._impl._data
            out = []
            for gk in group_keys:
                curr_max = None
                for i in groups[gk]:
                    v = raw[i]
                    if v is not None and (curr_max is None or v > curr_max):
                        curr_max = v
                out.append(curr_max)
            name = f"{col.name}_max" if col.name else "max"
            result_cols.append(PyVector(out, name=name))

        # STDEV (sample stdev, n-1)
        for col in stdev_cols:
            raw = col._impl._data
            out = []
            for gk in group_keys:
                vals = [raw[i] for i in groups[gk] if raw[i] is not None]
                m = len(vals)
                if m < 2:
                    out.append(None)
                    continue
                avg = sum(vals) / m
                var = sum((x - avg) ** 2 for x in vals) / (m - 1)
                out.append(math.sqrt(var))
            name = f"{col.name}_stdev" if col.name else "stdev"
            result_cols.append(PyVector(out, name=name))

        # CUSTOM APPLY
        if apply:
            for name, (col, func) in apply.items():
                raw = col._impl._data
                out = []
                for gk in group_keys:
                    subset = [raw[i] for i in groups[gk]]
                    out.append(func(subset))
                result_cols.append(PyVector(out, name=name))

        return PyTable(result_cols)

    def window(
        self,
        over,
        sum_over=None,
        mean_over=None,
        min_over=None,
        max_over=None,
        count_over=None,
        stdev_over=None,      # NEW
        apply=None,
    ):
        """
        Windowed aggregations:
        Groups by `over`, computes aggregates once per group,
        then broadcasts the result back to all rows in that group.
        """
        from .errors import PyVectorValueError
        import math

        # ---------------------------------------------------------
        # 1. Normalize partition keys
        # ---------------------------------------------------------
        if isinstance(over, PyVector):
            over = [over]

        nrows = self._length

        # Partition key length check
        for k in over:
            if len(k) != nrows:
                raise ValueError("Partition key length mismatch")

        # Utility to treat scalar-or-list aggregation args the same
        def _ensure_list(x):
            if x is None:
                return []
            return x if isinstance(x, list) else [x]

        sum_cols   = _ensure_list(sum_over)
        mean_cols  = _ensure_list(mean_over)
        min_cols   = _ensure_list(min_over)
        max_cols   = _ensure_list(max_over)
        count_cols = _ensure_list(count_over)
        stdev_cols = _ensure_list(stdev_over)

        # Collect aggregation columns to validate lengths
        agg_cols = sum_cols + mean_cols + min_cols + max_cols + count_cols + stdev_cols

        if apply:
            for col, _func in apply.values():
                agg_cols.append(col)

        # Aggregation column length checks
        for col in agg_cols:
            if len(col) != nrows:
                raise ValueError("Aggregation column length mismatch")

        # ---------------------------------------------------------
        # 2. Build groups (key → list of row indices)
        # ---------------------------------------------------------
        over_data = [c._impl._data for c in over]
        pk_len = len(over)

        groups = {}
        for idx in range(nrows):
            key = tuple(over_data[k][idx] for k in range(pk_len))
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)

        # ---------------------------------------------------------
        # 3. Setup for output
        # ---------------------------------------------------------
        result_cols = []

        def broadcast(indices, value, out):
            for i in indices:
                out[i] = value

        # ---------------------------------------------------------
        # 4. Aggregations
        # ---------------------------------------------------------

        # SUM
        for col in sum_cols:
            raw = col._impl._data
            out = [None] * nrows
            for indices in groups.values():
                total = 0
                for i in indices:
                    v = raw[i]
                    if v is not None:
                        total += v
                broadcast(indices, total, out)
            name = f"{col.name}_sum" if col.name else "sum"
            result_cols.append(PyVector(out, name=name))

        # COUNT
        for col in count_cols:
            raw = col._impl._data
            out = [None] * nrows
            for indices in groups.values():
                cnt = 0
                for i in indices:
                    if raw[i] is not None:
                        cnt += 1
                broadcast(indices, cnt, out)
            name = f"{col.name}_count" if col.name else "count"
            result_cols.append(PyVector(out, name=name))

        # MEAN
        for col in mean_cols:
            raw = col._impl._data
            out = [None] * nrows
            for indices in groups.values():
                total = 0
                cnt = 0
                for i in indices:
                    v = raw[i]
                    if v is not None:
                        total += v
                        cnt += 1
                val = total / cnt if cnt else None
                broadcast(indices, val, out)
            name = f"{col.name}_mean" if col.name else "mean"
            result_cols.append(PyVector(out, name=name))

        # MIN
        for col in min_cols:
            raw = col._impl._data
            out = [None] * nrows
            for indices in groups.values():
                curr_min = None
                for i in indices:
                    v = raw[i]
                    if v is not None and (curr_min is None or v < curr_min):
                        curr_min = v
                broadcast(indices, curr_min, out)
            name = f"{col.name}_min" if col.name else "min"
            result_cols.append(PyVector(out, name=name))

        # MAX
        for col in max_cols:
            raw = col._impl._data
            out = [None] * nrows
            for indices in groups.values():
                curr_max = None
                for i in indices:
                    v = raw[i]
                    if v is not None and (curr_max is None or v > curr_max):
                        curr_max = v
                broadcast(indices, curr_max, out)
            name = f"{col.name}_max" if col.name else "max"
            result_cols.append(PyVector(out, name=name))

        # STDEV (sample stdev)
        for col in stdev_cols:
            raw = col._impl._data
            out = [None] * nrows
            for indices in groups.values():
                vals = [raw[i] for i in indices if raw[i] is not None]
                m = len(vals)
                if m < 2:
                    val = None
                else:
                    avg = sum(vals) / m
                    var = sum((x - avg) ** 2 for x in vals) / (m - 1)
                    val = math.sqrt(var)
                broadcast(indices, val, out)
            name = f"{col.name}_stdev" if col.name else "stdev"
            result_cols.append(PyVector(out, name=name))

        # CUSTOM APPLY
        if apply:
            for name, (col, func) in apply.items():
                raw = col._impl._data
                out = [None] * nrows
                for indices in groups.values():
                    subset = [raw[i] for i in indices]
                    val = func(subset)
                    broadcast(indices, val, out)
                result_cols.append(PyVector(out, name=name))

        return PyTable(result_cols)
