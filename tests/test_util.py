import pytest
from damnit.util import complex2blob, blob2complex


@pytest.mark.parametrize("value", [
    1+2j,
    0+0j,
    -1.5-3.7j,
    2.5+0j,
    0+3.1j,
    float('inf')+0j,
    complex(float('inf'), -float('inf')),
])
def test_complex_blob_conversion(value):
    # Test that converting complex -> blob -> complex preserves the value
    blob = complex2blob(value)
    result = blob2complex(blob)
    assert result == value
