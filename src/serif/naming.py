"""Column name sanitization and uniquification utilities."""

from __future__ import annotations
import re


def _get_reserved_names():
	"""Get all public methods and properties from Vector and Table classes.
	
	This is computed dynamically to support future plugin extensions.
	Results are cached for performance.
	"""
	if not hasattr(_get_reserved_names, '_cache'):
		from .vector import Vector
		from .table import Table
		
		reserved = set()
		
		# Collect all public attributes from both classes
		for cls in (Vector, Table):
			for name in dir(cls):
				# Skip private/dunder attributes
				if name.startswith('_'):
					continue
				# Add public methods and properties
				attr = getattr(cls, name, None)
				if callable(attr) or isinstance(attr, property):
					reserved.add(name.lower())
		
		_get_reserved_names._cache = reserved
	
	return _get_reserved_names._cache


def _sanitize_user_name(name) -> str | None:
	"""Sanitize column name to valid Python identifier.
	
	Rules:
	- Convert to lowercase
	- Replace runs of non-alphanumeric chars (except _) with single _
	- Strip leading/trailing underscores
	- Prefix with 'c' if starts with digit
	- Append '_' if conflicts with reserved method names
	- Return None if empty after sanitization
	"""
	if not isinstance(name, str):
		name = str(name)
	
	# Lowercase
	name = name.lower()
	
	# Replace runs of invalid characters with _
	sanitized = re.sub(r'[^a-z0-9_]+', '_', name)
	
	# Strip leading/trailing _
	sanitized = sanitized.strip('_')
	
	# Empty → None
	if sanitized == "":
		return None
	
	# Starts with digit → prefix c
	if sanitized[0].isdigit():
		sanitized = "c" + sanitized
	
	# Conflicts with reserved name → append _
	if sanitized in _get_reserved_names():
		sanitized = sanitized + '_'
	
	return sanitized


def _uniquify(base: str, seen: set[str]) -> str:
	"""Make a unique name by adding __2, __3, etc if needed."""
	if base not in seen:
		return base
	
	i = 2
	while f"{base}__{i}" in seen:
		i += 1
	
	return f"{base}__{i}"


def resolve_column_name_for_binary_op(left_name, right_name):
	"""
	Apply left-biased naming rules for binary operations between columns.
	
	Rules:
	- If right is None or matches left: keep left (even if left is None)
	- If left is None but right is not: drop to None, return warning info
	- If both named but different: drop to None, return warning info
	
	Returns:
		tuple: (result_name, warning_case)
		       warning_case is None, "mismatch", or "right-named-left-unnamed"
	"""
	if right_name is None or right_name == left_name:
		# Keep left name (including if left is None)
		return (left_name, None)
	
	if left_name is None:
		# Case B: left unnamed, right named
		return (None, "right-named-left-unnamed")
	
	# Case A: both named but different
	return (None, "mismatch")
