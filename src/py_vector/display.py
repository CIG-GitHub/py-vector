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
	# Truncate with symmetric preview
	vals = col._underlying
	if len(vals) > max_preview * 2:
		preview = list(vals[:max_preview]) + ['...'] + list(vals[-max_preview:])
	else:
		preview = list(vals)

	# Type-sensitive formatting
	out = []
	for v in preview:
		if v == '...':
			out.append('...')
		elif col._dtype is float:
			out.append(f"{v:.1f}" if v == int(v) else f"{v:g}")
		elif col._dtype is int:
			out.append(str(v))
		elif col._dtype is date:
			out.append(v.isoformat())
		elif col._dtype is str:
			out.append(repr(v))
		else:
			out.append(str(v))

	# Align: numeric right, others left
	max_len = max(len(s) for s in out) if out else 0
	if col._dtype in (int, float):
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

		# Display name
		disp = col._name or ""
		display_names.append(disp)

		# Sanitized dot name
		if col._name:
			san = sanitize_func(col._name)
			if san is None:
				san = f"col{idx}_"
			else:
				san = uniquify_func(san, seen)
				seen.add(san)
		else:
			san = f"col{idx}_"
		sanitized_names.append(san)

		# Dtype
		dtypes.append(col._dtype.__name__ if col._dtype else "object")

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
	shape = pv.size()
	if not shape:
		return "# empty"
	
	if len(shape) == 1:
		dt = pv._dtype.__name__ if pv._dtype else "object"
		return f"# {len(pv)} element vector <{dt}>"
	
	if len(shape) == 2:
		if dtype_list:
			if truncated:
				d = ", ".join(dtype_list[:shown]) + ", ..., " + ", ".join(dtype_list[-shown:])
			else:
				d = ", ".join(dtype_list)
		else:
			d = pv._dtype.__name__ if pv._dtype else "object"
		rows, cols = shape
		return f"# {rows}×{cols} table <{d}>"
	
	shape_str = "×".join(str(s) for s in shape)
	dt = pv._dtype.__name__ if pv._dtype else "object"
	return f"# {shape_str} tensor <{dt}>"


def _repr_vector(v) -> str:
	"""Pretty repr for a 1D PyVector."""
	formatted = _format_column(v)
	
	# Compute width: max of data and header (if present)
	data_width = max(len(s) for s in formatted) if formatted else 0
	header_width = 0
	if v._name:
		header_text = repr(v._name) if _needs_quoting(v._name) else v._name
		header_width = len(header_text)
	
	width = max(data_width, header_width)
	
	# Re-align data to match combined width
	if v._dtype in (int, float):
		formatted = [s.rjust(width) for s in formatted]
	else:
		formatted = [s.ljust(width) for s in formatted]
	
	lines = []

	# Optional vector name
	if v._name:
		lines.append(header_text.ljust(width) if v._dtype not in (int, float) else header_text.rjust(width))

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
	dtypes_all = [col._dtype.__name__ if col._dtype else "object" for col in cols]

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
