"""Column name sanitization and uniquification utilities."""

from __future__ import annotations
import re


def _sanitize_user_name(name) -> str | None:
	"""Sanitize column name to valid Python identifier.
	
	Rules:
	- Convert to lowercase
	- Replace runs of non-alphanumeric chars (except _) with single _
	- Strip leading/trailing underscores
	- Prefix with 'c' if starts with digit
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
	
	return sanitized


def _uniquify(base: str, seen: set[str]) -> str:
	"""Make a unique name by adding __2, __3, etc if needed."""
	if base not in seen:
		return base
	
	i = 2
	while f"{base}__{i}" in seen:
		i += 1
	
	return f"{base}__{i}"
