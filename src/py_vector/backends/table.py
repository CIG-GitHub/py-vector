class TableBackend:
    """
    Backend for PyTable.

    Stores:
        - _data: tuple of raw Python tuples, one per column
        - _names: tuple[str | None]
        - _nrows: int
        - _name: table-level name (optional)
    """

    __slots__ = ("_data", "_names", "_nrows", "_name")

    def __init__(self, columns, names=None, nrows=None, name=None):
        """
        columns: tuple of raw column data (each is a Python tuple)
        names:   tuple of column names (may contain None)
        nrows:   number of rows (int)
        name:    name of the table (string or None)
        """

        # --- Validate input shape ---
        if not isinstance(columns, tuple):
            raise TypeError("TableBackend requires `columns` as a tuple")

        # Ensure each column is a tuple-like container
        for col in columns:
            if not hasattr(col, '__len__'):
                raise TypeError("Each column must be a sequence")

        # Row-count inference if not provided
        if nrows is None:
            nrows = len(columns[0]) if columns else 0

        # Final assignments
        self._data = columns
        self._names = names if names is not None else tuple(None for _ in columns)
        self._nrows = nrows
        self._name = name

    # ---------------------------------------------------------------------
    # Required public API for PyVector & PyTable wrappers
    # ---------------------------------------------------------------------

    def __len__(self):
        """Length = number of columns (vector semantics)."""
        return len(self._data)

    def __iter__(self):
        """Iterate raw columns."""
        return iter(self._data)

    def get_column(self, idx):
        """Return raw column tuple."""
        return self._data[idx]

    # ---------------------------------------------------------------------
    # Row indexing & slicing
    # ---------------------------------------------------------------------

    def __getitem__(self, key):
        """
        key:
            - int     -> return tuple of scalars (one per column)
            - slice   -> return new TableBackend slice
            - list    -> fancy index (mask or integer list)
        """
        # 1. Scalar row lookup
        if isinstance(key, int):
            return tuple(col[key] for col in self._data)

        # 2. Slice
        if isinstance(key, slice):
            sliced_cols = tuple(tuple(col[key]) for col in self._data)
            return TableBackend(
                sliced_cols,
                names=self._names,
                nrows=len(sliced_cols[0]) if sliced_cols else 0,
                name=self._name
            )

        # 3. Fancy indexing: list of row indices or booleans
        if isinstance(key, list):
            if not key:
                return TableBackend((), names=(), nrows=0, name=self._name)

            # Boolean mask
            if isinstance(key[0], bool):
                if len(key) != self._nrows:
                    raise ValueError("Mask length mismatch")
                indices = [i for i, flag in enumerate(key) if flag]
            else:
                # integer index list
                indices = key

            out = []
            for col in self._data:
                out.append(tuple(col[i] for i in indices))

            return TableBackend(
                tuple(out),
                names=self._names,
                nrows=len(indices),
                name=self._name
            )

        raise TypeError(f"Invalid index type for TableBackend: {type(key)}")

    # ---------------------------------------------------------------------
    # Fingerprint (for hashing tests)
    # ---------------------------------------------------------------------
    
    def fingerprint(self):
        """
        A fast, deterministic identity hash for the entire table.
        """
        h = 0x345678
        for col in self._data:
            h ^= hash(col) + 0x9e3779b9 + (h << 6) + (h >> 2)
        return h
