import urllib.request

from sstestkit.flakyhttp import FlakyHTTP
from tinyllm.demo.scale import scale


def test_testkit_reaches_course_tests():
    # WHY: course Python tests import sstestkit from the overlay's PYTHONPATH.
    # KIND: unit
    with FlakyHTTP({"/x": b"123"}) as srv:
        body = urllib.request.urlopen(srv.url("/x"), timeout=5).read()
    assert scale([float(c) for c in body.decode()], 1.0) == [1.0, 2.0, 3.0]
