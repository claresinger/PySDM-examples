import os
import pathlib
import re

import pytest


# https://stackoverflow.com/questions/7012921/recursive-grep-using-python
def findfiles(path, regex):
    reg_obj = re.compile(regex)
    res = []
    for root, _, fnames in os.walk(path):
        for fname in fnames:
            if reg_obj.match(fname):
                res.append(os.path.join(root, fname))
    return res


@pytest.fixture(
    params=findfiles(
        pathlib.Path(__file__).parent.parent.absolute().joinpath("PySDM_examples"),
        r".*\.(py)$",
    )
)
def example_filename(request):
    return request.param


# pylint: disable=redefined-outer-name
def test_run_examples(example_filename):
    if pathlib.Path(example_filename).name == "__init__.py":
        return
    with open(example_filename, encoding="utf8") as f:
        exec(f.read(), {"__name__": "__main__"})  # pylint: disable=exec-used
