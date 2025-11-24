from typing import Any, Dict, List, Tuple, Union

from .abstract import AbstractVector, VectorOpsMixin
from .errors import PyVectorTypeError, PyVectorValueError, PyVectorIndexError


class RowVector(VectorOpsMixin, AbstractVector):
    """Lightweight row view for table iteration with full vector operations."""
    
    __slots__ = ("_table", "_idx", "_name")
    
    def __init__(self, table: "PyTable", idx: int):
        self._table = table
        self._idx = idx
        self._name = None

    def __len__(self):
        return len(self._table._columns)

    def __getitem__(self, key: Union[int, str]) -> Any:
        cols = self._table._columns
        if isinstance(key, int):
            if not (0 <= key < len(cols)):
                raise PyVectorIndexError(
                    f"Column index {key} out of range for {len(cols)} columns"
                )
            return cols[key][self._idx]
        elif isinstance(key, str):
            col = self._table._name_to_col.get(key)
            if col is None:
                raise KeyError(f"Column {key!r} not found")
            return col[self._idx]
        else:
            raise PyVectorTypeError("Row indexing expects int or column name")

    def schema(self):
        return None

    def clone_with(self, data):
        raise TypeError("RowVector cannot be cloned")

    def _elementwise(self, func, other):
        raise TypeError("RowVector does not support elementwise operations")

    def _reduce(self, func):
        raise TypeError("RowVector does not support reductions")

    def to_dict(self) -> Dict[str, Any]:
        return {name: col[self._idx] for name, col in self._table._name_to_col.items()}

    def __repr__(self) -> str:
        return f"RowVector({self.to_dict()})"


class PyTable:
    """
    Columnar table holding multiple PyVectors.

    - No inheritance from PyVector (Bug #3, #4).
    - Composition only: _columns is a list/tuple of PyVector.
    """

    def __init__(self, initial: Any = None):
        """
        initial can be:
        - dict[str, Sequence or PyVector]
        - list[PyVector]
        - None / empty
        """
        self._columns: List[PyVector] = []
        self._name_to_col: Dict[str, PyVector] = {}
        self._length: int = 0

        if initial is None:
            return

        # Dict initialization: PyTable({'a': [...], 'b': [...]})
        if isinstance(initial, dict):
            for name, col_data in initial.items():
                if isinstance(col_data, PyVector):
                    col = col_data
                else:
                    col = PyVector(col_data, name=name)
                self._add_column(col)
        # List of columns: PyTable([PyVector(...), PyVector(...)])
        elif isinstance(initial, (list, tuple)):
            for col in initial:
                if not isinstance(col, PyVector):
                    raise PyVectorTypeError(
                        "PyTable(list) expects a list of PyVector objects"
                    )
                self._add_column(col)
        else:
            raise PyVectorTypeError(
                "PyTable initial must be dict, list[PyVector], or None"
            )

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _add_column(self, col: PyVector, name: str = None):
        """
        Internal: add a single column, enforcing length consistency
        and empty-table initialization semantics (Bug #18).
        """
        if name is None:
            name = getattr(col, "_name", None)

        if not name:
            # auto-generated name based on current column count
            base = f"col{len(self._columns)}"
            name = _sanitize_user_name(base)

        if self._length == 0:
            # Empty table: first column defines length
            self._length = len(col)
        else:
            if len(col) != self._length:
                raise PyVectorValueError(
                    f"New column length {len(col)} does not match table length {self._length}"
                )

        # Ensure unique name
        final_name = name
        counter = 2
        while final_name in self._name_to_col:
            final_name = f"{name}_{counter}"
            counter += 1

        # Set name on vector if it doesn't have one or conflicts
        if getattr(col, "_name", None) != final_name:
            # If your PyVector supports rename(), you can call that here.
            col = PyVector(col, name=final_name)

        self._columns.append(col)
        self._name_to_col[final_name] = col

    def _copy_with_columns(self, cols: List[PyVector]) -> "PyTable":
        t = PyTable()
        t._columns = list(cols)
        t._name_to_col = {getattr(c, "_name", f"col{i}"): c for i, c in enumerate(cols)}
        t._length = self._length if cols else 0
        return t

    # ------------------------------------------------------------------
    # Basic protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._length

    def cols(self) -> List[PyVector]:
        return list(self._columns)

    def names(self) -> List[str]:
        return [getattr(c, "_name", f"col{i}") for i, c in enumerate(self._columns)]

    def __iter__(self):
        for i in range(self._length):
            yield RowVector(self, i)

    # ------------------------------------------------------------------
    # Indexing: 1D and 2D (Bug #9: no recursive multi-dim)
    # ------------------------------------------------------------------

    def __getitem__(self, key: Any):
        # 2D indexing: table[row_sel, col_sel]
        if isinstance(key, tuple) and len(key) == 2:
            row_sel, col_sel = key
            return self._getitem_2d(row_sel, col_sel)

        # Column selection by name
        if isinstance(key, str):
            col = self._name_to_col.get(key)
            if col is None:
                raise KeyError(f"Column {key!r} not found")
            return col

        # Column selection by index
        if isinstance(key, int):
            if not (0 <= key < len(self._columns)):
                raise PyVectorIndexError(
                    f"Column index {key} out of range for {len(self._columns)} columns"
                )
            return self._columns[key]

        # List/tuple of columns → new PyTable with subset of columns
        if isinstance(key, (list, tuple)):
            cols_out: List[PyVector] = []
            for k in key:
                if isinstance(k, str):
                    col = self._name_to_col.get(k)
                    if col is None:
                        raise KeyError(f"Column {k!r} not found")
                    cols_out.append(col)
                elif isinstance(k, int):
                    if not (0 <= k < len(self._columns)):
                        raise PyVectorIndexError(
                            f"Column index {k} out of range for {len(self._columns)} columns"
                        )
                    cols_out.append(self._columns[k])
                else:
                    raise PyVectorTypeError(
                        "Column selection list must contain ints or strings"
                    )
            return self._copy_with_columns(cols_out)

        raise PyVectorTypeError("Invalid key type for PyTable.__getitem__")

    def _getitem_2d(self, row_sel: Any, col_sel: Any):
        """
        Explicit 2D indexing:
        - row_sel: int or slice
        - col_sel: str/int or list thereof
        """
        # Column subset
        if isinstance(col_sel, (str, int, list, tuple)):
            sub_table = self[col_sel]  # leverages 1D logic
            if isinstance(sub_table, PyVector):
                # Single-column table logically → use that as vector and slice rows
                return sub_table[row_sel]
            else:
                # It's a PyTable: now slice rows
                if isinstance(row_sel, int):
                    # Return a row view from the sub_table
                    if not (0 <= row_sel < len(sub_table)):
                        raise PyVectorIndexError(
                            f"Row index {row_sel} out of range for table length {len(sub_table)}"
                        )
                    return RowVector(sub_table, row_sel)
                elif isinstance(row_sel, slice):
                    # Row slice → new table with sliced columns
                    new_cols = [col[row_sel] for col in sub_table._columns]
                    t = PyTable(new_cols)
                    return t
                else:
                    raise PyVectorTypeError(
                        "Row selector must be int or slice in 2D indexing"
                    )
        else:
            raise PyVectorTypeError("Invalid column selector type in 2D indexing")

    # ------------------------------------------------------------------
    # Append columns: >> and >>= (Bug #10 semantics clarified)
    # ------------------------------------------------------------------

    def __rshift__(self, other: Any) -> "PyTable":
        """
        t >> col   → returns a new PyTable with column(s) appended.
        """
        new_table = PyTable()
        new_table._length = self._length
        new_table._columns = list(self._columns)
        new_table._name_to_col = dict(self._name_to_col)

        # Normalize 'other' into list of PyVector
        cols_to_add: List[PyVector] = []

        if isinstance(other, PyVector):
            cols_to_add.append(other)
        elif isinstance(other, (list, tuple)):
            # Could be list of PyVector or a single column worth of data
            if all(isinstance(c, PyVector) for c in other):
                cols_to_add.extend(other)
            else:
                # treat as a single new column of raw data
                cols_to_add.append(PyVector(other))
        else:
            # Single raw column
            cols_to_add.append(PyVector(other))

        for col in cols_to_add:
            new_table._add_column(col)

        return new_table

    def __irshift__(self, other: Any) -> "PyTable":
        """
        t >>= col  → mutates this table to append column(s).
        Copy-on-write-ish at table level (columns are immutable).
        """
        result = self >> other
        self._columns = result._columns
        self._name_to_col = result._name_to_col
        self._length = result._length
        return self

    # ------------------------------------------------------------------
    # Join key validation + type checks (Bug #8 + type sanity)
    # ------------------------------------------------------------------

    def _validate_join_keys(self, other: "PyTable", left_on, right_on):
        """
        Normalize and validate join keys.

        left_on / right_on:
        - str (column name)
        - PyVector
        - list of those

        Returns: list of (left_col, right_col)
        """

        def get_column(table: "PyTable", spec, side_name: str) -> PyVector:
            if isinstance(spec, str):
                col = table._name_to_col.get(spec)
                if col is None:
                    raise KeyError(f"Join key {spec!r} not found in {side_name} table")
                return col
            elif isinstance(spec, PyVector):
                return spec
            else:
                raise PyVectorValueError(
                    f"{side_name} join key must be str or PyVector, got {type(spec).__name__}"
                )

        # Normalize to lists
        if isinstance(left_on, (str, PyVector)):
            left_on = [left_on]
        if isinstance(right_on, (str, PyVector)):
            right_on = [right_on]

        if not isinstance(left_on, list) or not isinstance(right_on, list):
            raise PyVectorValueError("left_on and right_on must be strings, PyVectors, or lists")

        if len(left_on) == 0 or len(right_on) == 0:
            raise PyVectorValueError("Must specify at least one join key")

        if len(left_on) != len(right_on):
            raise PyVectorValueError(
                f"left_on and right_on must have same length: {len(left_on)} vs {len(right_on)}"
            )

        normalized = []
        for i, (l_spec, r_spec) in enumerate(zip(left_on, right_on)):
            lcol = get_column(self, l_spec, "Left")
            rcol = get_column(other, r_spec, "Right")

            if len(lcol) != len(self):
                raise PyVectorValueError(
                    f"Left join key at index {i} has length {len(lcol)}, "
                    f"but left table has {len(self)} rows"
                )
            if len(rcol) != len(other):
                raise PyVectorValueError(
                    f"Right join key at index {i} has length {len(rcol)}, "
                    f"but right table has {len(other)} rows"
                )

            # Optional: forbid float keys
            if lcol.schema() and lcol.schema().kind == float:
                raise PyVectorTypeError("Float columns are not allowed as join keys")
            if rcol.schema() and rcol.schema().kind == float:
                raise PyVectorTypeError("Float columns are not allowed as join keys")

            normalized.append((lcol, rcol))

        return normalized

    @staticmethod
    def _validate_key_tuple_hashable(key_tuple, key_cols, row_idx, side: str):
        # We only need this if schemas are object/None; here we assume keys are hashable.
        for i, comp in enumerate(key_tuple):
            try:
                hash(comp)
            except Exception:
                name = getattr(key_cols[i], "_name", f"key{i}")
                raise PyVectorTypeError(
                    f"Unhashable join key component at {side} row {row_idx}, column {name!r}"
                )

    # ------------------------------------------------------------------
    # Join implementations (left, inner, full)
    # ------------------------------------------------------------------

    def join(self, other: "PyTable", left_on, right_on, expect: str = "many_to_one") -> "PyTable":
        """
        Left join: keep all rows from left, match from right or fill with None.

        expect: 'one_to_one', 'many_to_one', 'one_to_many', 'many_to_many'
        """
        on = self._validate_join_keys(other, left_on, right_on)

        left_keys = [l for (l, _) in on]
        right_keys = [r for (_, r) in on]

        # Build right index: key_tuple -> [row_idx]
        right_index: Dict[Tuple[Any, ...], List[int]] = {}
        validate_hashable = True  # we can refine this later

        for row_idx in range(len(other)):
            key = tuple(rk[row_idx] for rk in right_keys)
            if validate_hashable:
                self._validate_key_tuple_hashable(key, right_keys, row_idx, side="right")
            right_index.setdefault(key, []).append(row_idx)

        # Cardinality check on right side
        if expect in ("one_to_one", "many_to_one"):
            dup = {k: idxs for k, idxs in right_index.items() if len(idxs) > 1}
            if dup:
                sample_key, sample_idxs = next(iter(dup.items()))
                raise PyVectorValueError(
                    f"Join expectation '{expect}' violated: Right side has duplicate keys. "
                    f"Example key {sample_key!r} occurs {len(sample_idxs)} times."
                )

        # Left-side uniqueness for one_to_one / one_to_many
        left_seen = set() if expect in ("one_to_one", "one_to_many") else None

        result_rows: List[List[Any]] = []
        for i in range(len(self)):
            key = tuple(lk[i] for lk in left_keys)
            if validate_hashable:
                self._validate_key_tuple_hashable(key, left_keys, i, side="left")

            if left_seen is not None:
                if key in left_seen:
                    raise PyVectorValueError(
                        f"Join expectation '{expect}' violated: Left side has duplicate key {key!r}"
                    )
                left_seen.add(key)

            if key in right_index:
                for r_idx in right_index[key]:
                    row = [col[i] for col in self._columns] + [col[r_idx] for col in other._columns]
                    result_rows.append(row)
            else:
                # no match: right side None
                row = [col[i] for col in self._columns] + [None] * len(other._columns)
                result_rows.append(row)

        if not result_rows:
            return PyTable([])

        # Construct result columns by zipping
        num_cols = len(self._columns) + len(other._columns)
        cols_out: List[PyVector] = []
        for col_idx in range(num_cols):
            col_data = [row[col_idx] for row in result_rows]
            if col_idx < len(self._columns):
                orig = self._columns[col_idx]
            else:
                orig = other._columns[col_idx - len(self._columns)]
            cols_out.append(PyVector(col_data, name=getattr(orig, "_name", None)))

        return PyTable(cols_out)

    def inner_join(self, other: "PyTable", left_on, right_on, expect: str = "many_to_one") -> "PyTable":
        """
        Inner join: only rows where keys match in both tables.
        """
        on = self._validate_join_keys(other, left_on, right_on)
        left_keys = [l for (l, _) in on]
        right_keys = [r for (_, r) in on]

        right_index: Dict[Tuple[Any, ...], List[int]] = {}
        validate_hashable = True

        for row_idx in range(len(other)):
            key = tuple(rk[row_idx] for rk in right_keys)
            if validate_hashable:
                self._validate_key_tuple_hashable(key, right_keys, row_idx, side="right")
            right_index.setdefault(key, []).append(row_idx)

        if expect in ("one_to_one", "many_to_one"):
            dup = {k: idxs for k, idxs in right_index.items() if len(idxs) > 1}
            if dup:
                sample_key, sample_idxs = next(iter(dup.items()))
                raise PyVectorValueError(
                    f"Join expectation '{expect}' violated: Right side has duplicate keys. "
                    f"Example key {sample_key!r} occurs {len(sample_idxs)} times."
                )

        left_seen = set() if expect in ("one_to_one", "one_to_many") else None

        result_rows: List[List[Any]] = []
        for i in range(len(self)):
            key = tuple(lk[i] for lk in left_keys)
            if validate_hashable:
                self._validate_key_tuple_hashable(key, left_keys, i, side="left")

            if left_seen is not None:
                if key in left_seen:
                    raise PyVectorValueError(
                        f"Join expectation '{expect}' violated: Left side has duplicate key {key!r}"
                    )
                left_seen.add(key)

            if key in right_index:
                for r_idx in right_index[key]:
                    row = [col[i] for col in self._columns] + [col[r_idx] for col in other._columns]
                    result_rows.append(row)

        if not result_rows:
            return PyTable([])

        num_cols = len(self._columns) + len(other._columns)
        cols_out: List[PyVector] = []
        for col_idx in range(num_cols):
            col_data = [row[col_idx] for row in result_rows]
            if col_idx < len(self._columns):
                orig = self._columns[col_idx]
            else:
                orig = other._columns[col_idx - len(self._columns)]
            cols_out.append(PyVector(col_data, name=getattr(orig, "_name", None)))

        return PyTable(cols_out)

    def full_join(self, other: "PyTable", left_on, right_on, expect: str = "many_to_many") -> "PyTable":
        """
        Full outer join: all rows from both tables.
        Unmatched rows are filled with None on the other side.
        """
        on = self._validate_join_keys(other, left_on, right_on)
        left_keys = [l for (l, _) in on]
        right_keys = [r for (_, r) in on]

        right_index: Dict[Tuple[Any, ...], List[int]] = {}
        validate_hashable = True

        for row_idx in range(len(other)):
            key = tuple(rk[row_idx] for rk in right_keys)
            if validate_hashable:
                self._validate_key_tuple_hashable(key, right_keys, row_idx, side="right")
            right_index.setdefault(key, []).append(row_idx)

        if expect in ("one_to_one", "many_to_one"):
            dup = {k: idxs for k, idxs in right_index.items() if len(idxs) > 1}
            if dup:
                sample_key, sample_idxs = next(iter(dup.items()))
                raise PyVectorValueError(
                    f"Join expectation '{expect}' violated: Right side has duplicate keys. "
                    f"Example key {sample_key!r} occurs {len(sample_idxs)} times."
                )

        left_seen = set() if expect in ("one_to_one", "one_to_many") else None
        matched_right_rows = set()

        result_rows: List[List[Any]] = []

        # Left side
        for i in range(len(self)):
            key = tuple(lk[i] for lk in left_keys)
            if validate_hashable:
                self._validate_key_tuple_hashable(key, left_keys, i, side="left")

            if left_seen is not None:
                if key in left_seen:
                    raise PyVectorValueError(
                        f"Join expectation '{expect}' violated: Left side has duplicate key {key!r}"
                    )
                left_seen.add(key)

            if key in right_index:
                for r_idx in right_index[key]:
                    matched_right_rows.add(r_idx)
                    row = [col[i] for col in self._columns] + [col[r_idx] for col in other._columns]
                    result_rows.append(row)
            else:
                row = [col[i] for col in self._columns] + [None] * len(other._columns)
                result_rows.append(row)

        # Unmatched right rows
        for r_idx in range(len(other)):
            if r_idx not in matched_right_rows:
                row = [None] * len(self._columns) + [col[r_idx] for col in other._columns]
                result_rows.append(row)

        if not result_rows:
            return PyTable([])

        num_cols = len(self._columns) + len(other._columns)
        cols_out: List[PyVector] = []
        for col_idx in range(num_cols):
            col_data = [row[col_idx] for row in result_rows]
            if col_idx < len(self._columns):
                orig = self._columns[col_idx]
            else:
                orig = other._columns[col_idx - len(self._columns)]
            cols_out.append(PyVector(col_data, name=getattr(orig, "_name", None)))

        return PyTable(cols_out)

    # ------------------------------------------------------------------
    # aggregate() / window() can be attached as methods here
    # (you already have good versions; we can plug them in next)
    # ------------------------------------------------------------------
