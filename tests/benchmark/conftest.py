import os

import pytest


def pytest_collection_modifyitems(items):
    benchmark_dir = os.path.join("tests", "benchmark")
    for item in items:
        if benchmark_dir in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
