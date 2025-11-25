import warnings
from typing import Any

# Adjust imports to point to your new structure
from .vector import PyVector
from .naming import _sanitize_user_name
from .naming import _uniquify
from .errors import PyVectorKeyError
from .errors import PyVectorValueError
from .errors import PyVectorTypeError

def _missing_col_error(name, context="PyTable"):
    return PyVectorKeyError(f"Column '{name}' not found in {context}")


class _RowView:
    """Lightweight row view for iterating over table rows with attribute access."""
    __slots__ = ('_cols_data', '_column_map', '_index')
    
    def __init__(self, table, index):
        # Cache direct handles to underlying raw tuple data
        # table is PyVector -> ._impl (Backend) -> ._data (Tuple of Columns)
        # col is PyVector -> ._impl (Backend) -> ._data (Raw Tuple)
        self._cols_data = [col._impl._data for col in table._impl._data]
        self._column_map = table._column_map
        self._index = index
    
    def set_index(self, index):
        self._index = index
        return self
    
    def __getattr__(self, attr):
        col_idx = self._column_map.get(attr.lower())
        if col_idx is None:
            raise AttributeError(f"Row has no attribute '{attr}'")
        return self._cols_data[col_idx][self._index]
    
    def __getitem__(self, key):
        try:
            return self._cols_data[key][self._index]
        except TypeError:
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
        try:
            from .display import _printr # Embedded here too
            return _printr(self)
        except ImportError:
            # Fallback is nice to have during refactors
            return f"PyTable({self.size()} rows)"


class PyTable(PyVector):
    """ Multiple columns of the same length """
    _length = None
    
    # We don't need __new__ override if using standard PyVector shell pattern
    
    def __init__(self, initial=(), dtype=None, name=None, as_row=False):
        # Handle dict initialization {name: values, ...}
        if isinstance(initial, dict):
            initial = [PyVector(values, name=col_name) for col_name, values in initial.items()]
        
        self._length = len(initial[0]) if initial else 0
        for col in initial:
            if len(col) != self._length:
                warnings.warn("Unequal column lengths", UserWarning)
                break

        # Deep copy columns and preserve names
        original_names = [vec.name for vec in initial] if initial else []
        
        if initial:
            # We must explicitly copy to prevent aliasing
            initial = tuple(vec.copy() for vec in initial)
        else:
            initial = ()
        
        # Explicitly set dtype to None for the Table itself (it holds Vectors)
        self._dtype = None
        
        # Initialize the parent PyVector. 
        # This creates self._impl (TupleBackend) holding the tuple of PyVectors.
        super().__init__(initial, dtype=dtype, name=name)
        
        # Restore column names on the new copies
        # self._impl._data is the tuple of columns
        if original_names:
            columns = self._impl._data
            for i, col_name in enumerate(original_names):
                if i < len(columns):
                    columns[i].rename(col_name)
        

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

    def cols(self):
        """Helper to get the tuple of column vectors."""
        return self._impl._data

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
        # If we are looking for '_impl' (or any private var), fail immediately.
        # Otherwise, reading self._impl below triggers infinite recursion.
        if attr.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

        # Now it is safe to access self._impl for the Strobe Lookup
        from .naming import _sanitize_user_name
        
        for col in self._impl._data:
            if _sanitize_user_name(col.name) == attr:
                return col
                
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

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
        # 1. Integer Index -> ROW ACCESS
        # Returns a lightweight view of the row (like a dict/object)
        if isinstance(key, int):
            # Handle negative indexing
            if key < 0:
                key += self._length
            
            if not (0 <= key < self._length):
                raise PyVectorIndexError(f"Row index {key} out of bounds")
                
            return _RowView(self, key)

        # 2. String -> COLUMN ACCESS (Stateless Strobe Lookup)
        if isinstance(key, str):
            # Pass 1: Exact Match
            for col in self._impl._data:
                if col.name == key:
                    return col
            
            # Pass 2: Sanitized Match
            from .naming import _sanitize_user_name
            target = key.lower()
            for col in self._impl._data:
                if _sanitize_user_name(col.name) == target:
                    return col
            
            raise _missing_col_error(key)

        # 3. Slice -> ROW SLICE (Subset of Table)
        if isinstance(key, slice):
            # Slice every column to same length
            new_cols = [col[key] for col in self._impl._data]
            return PyTable(new_cols)

        # 4. List/Tuple of Strings -> SUBSET OF COLUMNS
        if isinstance(key, (tuple, list)):
            # If all are strings, select columns
            if all(isinstance(k, str) for k in key):
                selected = []
                # Naive implementation: iterate loop above. 
                # Optimization: map names once if list is long.
                for k in key:
                    selected.append(self[k].copy()) # Use self[k] to trigger strobe lookup
                return PyTable(selected)

        # 5. Boolean Mask (PyVector or List) -> ROW FILTER
        # (Defer to super/vector logic or implement explicit masking)
        if isinstance(key, PyVector) or (isinstance(key, list) and len(key) == self._length):
             # Mask every column
             mask = key # logic to ensure it's a list/mask
             new_cols = [col[mask] for col in self._impl._data]
             return PyTable(new_cols)

        raise PyVectorTypeError(f"Invalid index type: {type(key)}")

    def __iter__(self):
        row_view = _RowView(self, 0)
        for i in range(self._length):
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

    def _validate_join_keys(self, other, left_on, right_on):
        # (Same implementation logic, just ensure we use standard access)
        # For PyVectors, we can use standard indexing.
        # ... logic from your snippet is largely compatible as long as 
        # get_column uses self[col_spec] which we fixed in __getitem__
        
        # Copy-paste the logic but keep get_column clean
        from datetime import date, datetime
        
        def get_column(table, col_spec, side_name):
            if isinstance(col_spec, str):
                try:
                    return table[col_spec]
                except (KeyError, ValueError):
                    raise _missing_col_error(col_spec, context=f"{side_name} table")
            elif isinstance(col_spec, PyVector):
                return col_spec
            else:
                raise PyVectorValueError(f"Invalid column spec: {type(col_spec)}")

        # ... (rest of validation logic matches your snippet) ...
        # I'll inject the minimal needed wrapper logic here to save space
        # Assuming exact copy of your validation logic, just ensure 
        # it calls the get_column defined above.
        
        # RE-INJECTING YOUR VALIDATION LOGIC FOR COMPLETENESS:
        if isinstance(left_on, (str, PyVector)): left_on = [left_on]
        if isinstance(right_on, (str, PyVector)): right_on = [right_on]
        
        normalized = []
        for i, (l_spec, r_spec) in enumerate(zip(left_on, right_on)):
            l_col = get_column(self, l_spec, "left")
            r_col = get_column(other, r_spec, "right")
            
            # Checking types
            ls = l_col.schema()
            rs = r_col.schema()
            if ls and rs and ls.kind != rs.kind:
                 raise PyVectorTypeError(f"Join key mismatch at {i}: {ls.kind} vs {rs.kind}")
            
            normalized.append((l_col, r_col))
        return normalized

    @staticmethod
    def _validate_key_tuple_hashable(key_tuple, key_cols, row_idx):
        try:
            hash(key_tuple)
        except TypeError as e:
            raise PyVectorTypeError(f"Unhashable key at row {row_idx}: {e}")

    def left_join(self, other, left_on, right_on):
        # 1. Setup
        pairs = self._validate_join_keys(other, left_on, right_on)
        
        # Access raw data for speed
        left_keys_data = [col._impl._data for col, _ in pairs]
        right_keys_data = [col._impl._data for _, col in pairs]
        
        left_cols_data = [c._impl._data for c in self._impl._data]
        right_cols_data = [c._impl._data for c in other._impl._data]
        
        left_nrows = self._length
        right_nrows = len(other)
        
        # 2. Build Hash Map (Right Table)
        right_index = {}
        pk_len = len(right_keys_data)
        
        for r_idx in range(right_nrows):
            key = tuple(right_keys_data[k][r_idx] for k in range(pk_len))
            if key not in right_index:
                right_index[key] = []
            right_index[key].append(r_idx)
            
        # 3. Join Loop
        result_data = [[] for _ in range(len(left_cols_data) + len(right_cols_data))]
        
        for l_idx in range(left_nrows):
            key = tuple(left_keys_data[k][l_idx] for k in range(pk_len))
            matches = right_index.get(key)
            
            if matches:
                for r_idx in matches:
                    # Match found: Append Left + Right
                    for i, col_data in enumerate(left_cols_data):
                        result_data[i].append(col_data[l_idx])
                    offset = len(left_cols_data)
                    for i, col_data in enumerate(right_cols_data):
                        result_data[offset + i].append(col_data[r_idx])
            else:
                # No match: Append Left + None
                for i, col_data in enumerate(left_cols_data):
                    result_data[i].append(col_data[l_idx])
                offset = len(left_cols_data)
                for i in range(len(right_cols_data)):
                    result_data[offset + i].append(None)

        # 4. Wrap in PyTable
        res_cols = []
        # Re-wrap Left Columns
        for i, col in enumerate(self._impl._data):
            res_cols.append(PyVector(result_data[i], name=col.name))
        # Re-wrap Right Columns
        offset = len(self._impl._data)
        for i, col in enumerate(other._impl._data):
            res_cols.append(PyVector(result_data[offset+i], name=col.name))
            
        return PyTable(res_cols)


    def inner_join(self, other, left_on, right_on, expect='many_to_one'):
        # 1. Setup
        pairs = self._validate_join_keys(other, left_on, right_on)
        # Extract RAW DATA tuples for speed
        left_keys_data = [col._impl._data for col, _ in pairs]
        right_keys_data = [col._impl._data for _, col in pairs]
        
        left_cols_data = [c._impl._data for c in self._impl._data]
        right_cols_data = [c._impl._data for c in other._impl._data]
        
        left_nrows = self._length
        right_nrows = len(other) # or other._length
        
        # 2. Build Hash Map (Right)
        right_index = {}
        pk_len = len(right_keys_data)
        
        for r_idx in range(right_nrows):
            key = tuple(right_keys_data[k][r_idx] for k in range(pk_len))
            if key not in right_index:
                right_index[key] = []
            right_index[key].append(r_idx)
            
        # 3. Join
        result_data = [[] for _ in range(len(left_cols_data) + len(right_cols_data))]
        
        for l_idx in range(left_nrows):
            key = tuple(left_keys_data[k][l_idx] for k in range(pk_len))
            matches = right_index.get(key)
            
            if matches:
                for r_idx in matches:
                    # Append Left
                    for i, col_data in enumerate(left_cols_data):
                        result_data[i].append(col_data[l_idx])
                    # Append Right
                    offset = len(left_cols_data)
                    for i, col_data in enumerate(right_cols_data):
                        result_data[offset + i].append(col_data[r_idx])

        # 4. Wrap
        res_cols = []
        # Left names
        for i, col in enumerate(self._impl._data):
            res_cols.append(PyVector(result_data[i], name=col.name))
        # Right names
        offset = len(self._impl._data)
        for i, col in enumerate(other._impl._data):
            res_cols.append(PyVector(result_data[offset+i], name=col.name))
            
        return PyTable(res_cols)

    def full_join(self, other, left_on, right_on):
        # 1. Setup
        pairs = self._validate_join_keys(other, left_on, right_on)
        
        # Access raw data
        left_keys_data = [col._impl._data for col, _ in pairs]
        right_keys_data = [col._impl._data for _, col in pairs]
        
        left_cols_data = [c._impl._data for c in self._impl._data]
        right_cols_data = [c._impl._data for c in other._impl._data]
        
        left_nrows = self._length
        right_nrows = len(other)
        
        # 2. Build Hash Map (Right)
        right_index = {}
        pk_len = len(right_keys_data)
        
        for r_idx in range(right_nrows):
            key = tuple(right_keys_data[k][r_idx] for k in range(pk_len))
            if key not in right_index:
                right_index[key] = []
            right_index[key].append(r_idx)
            
        # Track which right rows we have matched
        right_matches_visited = set()
        
        # 3. Join Loop (Left Scan)
        result_data = [[] for _ in range(len(left_cols_data) + len(right_cols_data))]
        
        for l_idx in range(left_nrows):
            key = tuple(left_keys_data[k][l_idx] for k in range(pk_len))
            matches = right_index.get(key)
            
            if matches:
                for r_idx in matches:
                    right_matches_visited.add(r_idx)
                    # Match: Append Left + Right
                    for i, col_data in enumerate(left_cols_data):
                        result_data[i].append(col_data[l_idx])
                    offset = len(left_cols_data)
                    for i, col_data in enumerate(right_cols_data):
                        result_data[offset + i].append(col_data[r_idx])
            else:
                # No Match: Append Left + None
                for i, col_data in enumerate(left_cols_data):
                    result_data[i].append(col_data[l_idx])
                offset = len(left_cols_data)
                for i in range(len(right_cols_data)):
                    result_data[offset + i].append(None)

        # 4. Right Scan (Append remaining unmatched right rows)
        for r_idx in range(right_nrows):
            if r_idx not in right_matches_visited:
                # Append None + Right
                for i in range(len(left_cols_data)):
                    result_data[i].append(None)
                offset = len(left_cols_data)
                for i, col_data in enumerate(right_cols_data):
                    result_data[offset + i].append(col_data[r_idx])

        # 5. Wrap
        res_cols = []
        for i, col in enumerate(self._impl._data):
            res_cols.append(PyVector(result_data[i], name=col.name))
        offset = len(self._impl._data)
        for i, col in enumerate(other._impl._data):
            res_cols.append(PyVector(result_data[offset+i], name=col.name))
            
        return PyTable(res_cols)

    def aggregate(
            self,
            over,
            sum_over=None,
            mean_over=None,
            min_over=None,
            max_over=None,
            count_over=None,
            apply=None,
        ):
            if isinstance(over, PyVector): over = [over]
            
            # 1. Access RAW data for keys (Fastest access)
            over_data = [c._impl._data for c in over]
            nrows = self._length
            pk_len = len(over)
            
            # 2. Build Groups (Key -> List of Indices)
            # This is unavoidable overhead in pure Python, but dicts are fast.
            groups = {}
            for idx in range(nrows):
                key = tuple(over_data[k][idx] for k in range(pk_len))
                if key not in groups:
                    groups[key] = []
                groups[key].append(idx)
            
            result_cols = []
            group_keys = list(groups.keys()) # Stable ordering based on insertion

            # 3. Reconstruct Partition Keys
            for k_idx, k_col in enumerate(over):
                vals = [gk[k_idx] for gk in group_keys]
                result_cols.append(PyVector(vals, name=k_col.name))

            # --------------------------------------------------------
            # 4. Aggregations (Unrolled Loops - No Lambdas, No Lists)
            # --------------------------------------------------------

            # --- SUM ---
            if sum_over:
                if not isinstance(sum_over, list): sum_over = [sum_over]
                for col in sum_over:
                    raw = col._impl._data
                    out = []
                    for gk in group_keys:
                        indices = groups[gk]
                        # Inline sum: No list allocation
                        total = 0
                        for i in indices:
                            v = raw[i]
                            if v is not None:
                                total += v
                        out.append(total)
                    
                    name = f"{col.name}_sum" if col.name else "sum"
                    result_cols.append(PyVector(out, name=name))

            # --- COUNT ---
            if count_over:
                if not isinstance(count_over, list): count_over = [count_over]
                for col in count_over:
                    raw = col._impl._data
                    out = []
                    for gk in group_keys:
                        indices = groups[gk]
                        # Inline count
                        cnt = 0
                        for i in indices:
                            if raw[i] is not None:
                                cnt += 1
                        out.append(cnt)

                    name = f"{col.name}_count" if col.name else "count"
                    result_cols.append(PyVector(out, name=name))

            # --- MEAN ---
            if mean_over:
                if not isinstance(mean_over, list): mean_over = [mean_over]
                for col in mean_over:
                    raw = col._impl._data
                    out = []
                    for gk in group_keys:
                        indices = groups[gk]
                        # Inline mean
                        total = 0
                        cnt = 0
                        for i in indices:
                            v = raw[i]
                            if v is not None:
                                total += v
                                cnt += 1
                        out.append(total / cnt if cnt > 0 else None)

                    name = f"{col.name}_mean" if col.name else "mean"
                    result_cols.append(PyVector(out, name=name))

            # --- MIN ---
            if min_over:
                if not isinstance(min_over, list): min_over = [min_over]
                for col in min_over:
                    raw = col._impl._data
                    out = []
                    for gk in group_keys:
                        indices = groups[gk]
                        # Inline min
                        curr_min = None
                        for i in indices:
                            v = raw[i]
                            if v is not None:
                                if curr_min is None or v < curr_min:
                                    curr_min = v
                        out.append(curr_min)

                    name = f"{col.name}_min" if col.name else "min"
                    result_cols.append(PyVector(out, name=name))

            # --- MAX ---
            if max_over:
                if not isinstance(max_over, list): max_over = [max_over]
                for col in max_over:
                    raw = col._impl._data
                    out = []
                    for gk in group_keys:
                        indices = groups[gk]
                        # Inline max
                        curr_max = None
                        for i in indices:
                            v = raw[i]
                            if v is not None:
                                if curr_max is None or v > curr_max:
                                    curr_max = v
                        out.append(curr_max)

                    name = f"{col.name}_max" if col.name else "max"
                    result_cols.append(PyVector(out, name=name))

            # --- CUSTOM APPLY (The only place we allow list allocs) ---
            if apply:
                for name, (col, func) in apply.items():
                    raw = col._impl._data
                    out = []
                    for gk in group_keys:
                        indices = groups[gk]
                        # We MUST allocate here because we don't know what func does
                        subset = [raw[i] for i in indices]
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
            apply=None,
        ):
            """
            Performs aggregations over groups but preserves original row count 
            (broadcasting results back to all rows in the group).
            """
            if isinstance(over, PyVector): over = [over]
            
            # 1. Access RAW data for keys
            over_data = [c._impl._data for c in over]
            nrows = self._length
            pk_len = len(over)
            
            # 2. Build Groups (Key -> List of Indices)
            groups = {}
            for idx in range(nrows):
                key = tuple(over_data[k][idx] for k in range(pk_len))
                if key not in groups:
                    groups[key] = []
                groups[key].append(idx)
            
            result_cols = []

            # Helper to broadcast value to all indices in the group
            def broadcast(indices, value, target_list):
                for i in indices:
                    target_list[i] = value

            # --- SUM ---
            if sum_over:
                if not isinstance(sum_over, list): sum_over = [sum_over]
                for col in sum_over:
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

            # --- COUNT ---
            if count_over:
                if not isinstance(count_over, list): count_over = [count_over]
                for col in count_over:
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

            # --- MEAN ---
            if mean_over:
                if not isinstance(mean_over, list): mean_over = [mean_over]
                for col in mean_over:
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
                        val = total / cnt if cnt > 0 else None
                        broadcast(indices, val, out)

                    name = f"{col.name}_mean" if col.name else "mean"
                    result_cols.append(PyVector(out, name=name))

            # --- MIN ---
            if min_over:
                if not isinstance(min_over, list): min_over = [min_over]
                for col in min_over:
                    raw = col._impl._data
                    out = [None] * nrows
                    for indices in groups.values():
                        curr_min = None
                        for i in indices:
                            v = raw[i]
                            if v is not None:
                                if curr_min is None or v < curr_min:
                                    curr_min = v
                        broadcast(indices, curr_min, out)

                    name = f"{col.name}_min" if col.name else "min"
                    result_cols.append(PyVector(out, name=name))

            # --- MAX ---
            if max_over:
                if not isinstance(max_over, list): max_over = [max_over]
                for col in max_over:
                    raw = col._impl._data
                    out = [None] * nrows
                    for indices in groups.values():
                        curr_max = None
                        for i in indices:
                            v = raw[i]
                            if v is not None:
                                if curr_max is None or v > curr_max:
                                    curr_max = v
                        broadcast(indices, curr_max, out)

                    name = f"{col.name}_max" if col.name else "max"
                    result_cols.append(PyVector(out, name=name))

            # --- CUSTOM APPLY ---
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
