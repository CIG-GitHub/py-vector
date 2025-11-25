"""Display and repr logic for PyVector and PyTable."""

from __future__ import annotations
from datetime import date
from typing import List


# How many rows/columns to show before inserting "..."
MAX_HEAD_ROWS = 5
MAX_HEAD_COLS = 5


def _needs_quoting(name: str) -> bool:
	"""A name needs quoting if it contains anything outside [A-Za-z0-9_]
	OR has leading/trailing whitespace."""
	if not name:
		return False
	if name != name.strip():
		return True
	return not all(c.isalnum() or c == "_" for c in name)


def _format_column(col, max_preview: int = MAX_HEAD_ROWS) -> List[str]:
    """Returns a list of strings representing that column, truncated for display."""
    # 1. Use Public API for data access (Slicing/Iteration)
    # col is a PyVector, so it supports len() and slicing
    N = len(col)
    if N > max_preview * 2:
        # Slicing returns PyVector, cast to list for display processing
        head = list(col[:max_preview])
        tail = list(col[N - max_preview:])
        preview = head + ['...'] + tail
    else:
        preview = list(col)

    # 2. Use Public API for Metadata
    dtype = col.schema()
    
    # Type-sensitive formatting
    out = []
    for v in preview:
        if v == '...':
            out.append('...')
        elif v is None:
            out.append("None") # or "" depending on preference
        elif dtype and dtype.kind is float:
            # Handle standard float formatting
            out.append(f"{v:.1f}" if v == int(v) else f"{v:g}")
        elif dtype and dtype.kind is int:
            out.append(str(v))
        elif dtype and dtype.kind is date:
            out.append(v.isoformat())
        elif dtype and dtype.kind is str:
            out.append(repr(v))
        else:
            out.append(str(v))

    # Align: numeric right, others left
    max_len = max(len(s) for s in out) if out else 0
    if dtype and dtype.kind in (int, float):
        return [s.rjust(max_len) for s in out]
    return [s.ljust(max_len) for s in out]


def _compute_headers(cols, col_indices, sanitize_func, uniquify_func):
    """Given PyTable columns and indices, returns display_names, sanitized_names, dtypes."""
    display_names = []
    sanitized_names = []
    dtypes = []
    seen = set()

    for idx in col_indices:
        col = cols[idx]

        # Use Public API .name
        disp = col.name or ""
        display_names.append(disp)

        # Sanitized dot name
        if col.name:
            san = sanitize_func(col.name)
            if san is None:
                san = f"col{idx}_"
            else:
                san = uniquify_func(san, seen)
                seen.add(san)
        else:
            san = f"col{idx}_"
        sanitized_names.append(san)

        # Use Public API .schema()
        dt = col.schema()
        dtypes.append(dt.kind.__name__ if dt else "object")

    return display_names, sanitized_names, dtypes


def _header_rows(display_names, sanitized_names):
	"""Decide which header rows to show based on display vs sanitized names."""
	any_display = any(n for n in display_names if n != "...")
	any_mismatch = any(
		disp and san and disp != san and san != "..."
		for disp, san in zip(display_names, sanitized_names)
	)

	rows = []

	# Row 1: display names (quoted if needed)
	if any_display:
		row = []
		for name in display_names:
			if name == "...":
				row.append("...")
			elif _needs_quoting(name):
				row.append(repr(name))
			else:
				row.append(name if name else "")
		rows.append(row)

	# Row 2: sanitized names (if mismatch or no display names)
	if any_mismatch or not any_display:
		rows.append([("." + san) if san and san != "..." else san for san in sanitized_names])

	return rows


def _align_columns(formatted_cols, header_rows, col_dtypes):
	"""Pad columns and headers to consistent widths."""
	num_cols = len(formatted_cols)
	col_widths = []

	# Compute desired width per column
	for c in range(num_cols):
		body_width = max(len(s) for s in formatted_cols[c]) if formatted_cols[c] else 0
		header_width = max(
			len(header_rows[r][c]) for r in range(len(header_rows))
		) if header_rows else 0
		col_widths.append(max(body_width, header_width))

	# Re-pad columns based on dtype
	aligned_cols = []
	for c in range(num_cols):
		col = formatted_cols[c]
		w = col_widths[c]
		dtype = col_dtypes[c]
		if dtype in ('int', 'float'):
			aligned_cols.append([s.rjust(w) for s in col])
		else:
			aligned_cols.append([s.ljust(w) for s in col])

	# Re-pad headers
	aligned_headers = []
	for row in header_rows:
		aligned_headers.append([h.rjust(col_widths[c]) if col_dtypes[c] in ('int', 'float') else h.ljust(col_widths[c]) 
		                        for c, h in enumerate(row)])

	return aligned_cols, aligned_headers


def _footer(pv, dtype_list=None, truncated=False, shown=MAX_HEAD_COLS) -> str:
    """Generate footer line based on shape and dtypes."""
    # Use Public API .size()
    shape = pv.size()
    if not shape or shape == (0,):
        return "# empty"
    
    # Use Public API .schema()
    dtype = pv.schema()

    if len(shape) == 1:
        if dtype:
            dt = dtype.kind.__name__
            if dtype.nullable:
                dt += "?"
        else:
            dt = "object"
        return f"# {len(pv)} element vector <{dt}>"
    
    if len(shape) == 2:
        if dtype_list:
            if truncated:
                d = ", ".join(dtype_list[:shown]) + ", ..., " + ", ".join(dtype_list[-shown:])
            else:
                d = ", ".join(dtype_list)
        else:
            d = dtype.kind.__name__ if dtype else "object"
        rows, cols = shape
        return f"# {rows}×{cols} table <{d}>"
    
    # Fallback for ndims > 2
    return "# tensor"


def _repr_vector(v) -> str:
    """Pretty repr for a 1D PyVector."""
    formatted = _format_column(v)
    
    # Compute width
    data_width = max(len(s) for s in formatted) if formatted else 0
    header_width = 0
    
    # Use Public API .name
    if v.name:
        header_text = repr(v.name) if _needs_quoting(v.name) else v.name
        header_width = len(header_text)
    
    width = max(data_width, header_width)
    
    # Align
    dtype = v.schema()
    if dtype and dtype.kind in (int, float):
        formatted = [s.rjust(width) for s in formatted]
    else:
        formatted = [s.ljust(width) for s in formatted]
    
    lines = []

    # Optional vector name
    if v.name:
        lines.append(header_text.ljust(width) if not dtype or dtype.kind not in (int, float) else header_text.rjust(width))

    lines.extend(formatted)
    lines.append("")
    lines.append(_footer(v))
    return "\n".join(lines)


def _repr_table(tbl) -> str:
	"""Pretty repr for a 2D PyTable."""
	from .naming import _sanitize_user_name, _uniquify
	
	cols = tbl.cols()
	num_cols = len(cols)

	if num_cols == 0:
		return "# 0×0 table"

	truncated = num_cols > MAX_HEAD_COLS * 2

	if truncated:
		col_indices = list(range(MAX_HEAD_COLS)) + list(range(num_cols - MAX_HEAD_COLS, num_cols))
	else:
		col_indices = list(range(num_cols))

	# Headers + dtypes
	disp, san, dtypes_displayed = _compute_headers(
		cols, col_indices, _sanitize_user_name, _uniquify
	)

	# Get all dtypes for footer
	dtypes_all = []
	for col in cols:
		if col._dtype:
			dtype_str = col._dtype.kind.__name__
			if col._dtype.nullable:
				dtype_str += "?"
			dtypes_all.append(dtype_str)
		else:
			dtypes_all.append("object")

	# Format columns
	formatted_cols = [_format_column(cols[i]) for i in col_indices]

	# Insert "..." column if truncated
	if truncated:
		ellipsis_col = ["..." for _ in range(len(formatted_cols[0]))]
		formatted_cols.insert(MAX_HEAD_COLS, ellipsis_col)
		disp.insert(MAX_HEAD_COLS, "...")
		san.insert(MAX_HEAD_COLS, "...")
		dtypes_displayed.insert(MAX_HEAD_COLS, "...")

	# Build header rows
	header_rows = _header_rows(disp, san)

	# Align everything
	aligned_cols, aligned_headers = _align_columns(formatted_cols, header_rows, dtypes_displayed)

	# Build output
	lines = []
	for hrow in aligned_headers:
		lines.append("  ".join(hrow))

	# Table body
	nrows = len(aligned_cols[0]) if aligned_cols else 0
	for r in range(nrows):
		row = "  ".join(col[r] for col in aligned_cols)
		lines.append(row)

	lines.append("")
	lines.append(_footer(tbl, dtypes_all, truncated, MAX_HEAD_COLS))

	return "\n".join(lines)


def _printr(pv) -> str:
	"""Entry point used by PyVector.__repr__ and PyTable.__repr__."""
	nd = len(pv.size())
	if nd == 1:
		return _repr_vector(pv)
	if nd == 2:
		return _repr_table(pv)
	return _footer(pv) + " (repr not yet implemented)"
