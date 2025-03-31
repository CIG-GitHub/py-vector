import pytest

from pyvector import PyVector


@pytest.mark.parametrize('initial,default_elemnt,length, dtype,typesafe', [
    ([1, 2, 3], 0, int, True),
    ([1, 2, 3], 4, float, True),
    ([1, 2, 3], None, float, True),
    ([1, 2, 3], 2, None, True),
    ([], 2, None, True),
])
def test_initial(currency_iso, lookup_id):
    """ test that parsed currency isos return the correct id """
    pass



@pytest.mark.parametrize('initial,default_elemnt,dtype,typesafe', [
    ([1, 2, 3], 0, str, True),
    ([1, 2, 3], 4, str, True),
    ([1, 2, 3], None, str, True),
    (None, '2', int, True),
    ([], 2, str, True),
    ('', 2, str, True),
    ('', 2, str, True),
    ([1, 'bad', 3], None, int, True),
    ([1, True, 3], None, int, True), # Fail due to boolean
])