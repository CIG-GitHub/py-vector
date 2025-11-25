import pytest
from datetime import date
from py_vector import PyVector


def test_date_vector_creation():
    v = PyVector(["2024-01-02", "2024-03-05"], dtype=date)
    assert v.schema().kind is date
    assert isinstance(v[0], date)
    assert isinstance(v[1], date)
    assert v[0] == date(2024, 1, 2)


def test_date_casting_from_strings():
    v = PyVector(["2023-01-01", "2024-02-03"]).cast(date)
    assert v[0] == date(2023, 1, 1)
    assert v[1] == date(2024, 2, 3)


def test_date_preserves_existing_dates():
    v = PyVector([date(2023, 2, 3), date(2024, 4, 5)], dtype=date)
    v2 = v.cast(date)  # should NOT attempt to fromisoformat
    assert v2[0] == date(2023, 2, 3)
    assert v2[1] == date(2024, 4, 5)


def test_date_comparison():
    v = PyVector([date(2024, 1, 1), date(2023, 1, 1)])
    assert (v > date(2023, 12, 31)).all()
    assert (v < date(2024, 1, 2)).all()


def test_date_method_proxy():
    """Ensure v.year, v.month produce PyVector results."""
    v = PyVector(["2024-01-02", "2023-12-30"], dtype=date)
    years = v.year()
    months = v.month()
    assert years.schema().kind is int
    assert months.schema().kind is int
    assert years[0] == 2024
    assert months[1] == 12


def test_date_fillna():
    v = PyVector([date(2024, 1, 2), None], dtype=date)
    filled = v.fillna(date(2000, 1, 1))
    assert filled[1] == date(2000, 1, 1)


def test_date_unique():
    v = PyVector([date(2024,1,1), date(2024,1,1), date(2023,1,1)], dtype=date)
    u = v.unique()
    assert len(u) == 2
    assert date(2024,1,1) in u._backend._storage
    assert date(2023,1,1) in u._backend._storage
