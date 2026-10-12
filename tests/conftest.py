"""Pytest configuration."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run every test, even slow ones.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: Mark a test or option as slow to run")
    config.addinivalue_line("markers", "dependencies: Mark a test that needs optional dependencies")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        # --slow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="skipped, --slow not selected")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session", autouse=True)
def _run_from_tests_directory():
    """Run the suite from the tests directory.

    The chemistry tests migrated from tequila address their fixtures with paths
    relative to the tests directory (e.g. "data/h2.xyz") and also write
    scratch files there. CI already does `cd tests` before calling pytest; this
    keeps `pytest tests/` from the repository root working as well.
    """
    import os

    previous = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        yield
    finally:
        os.chdir(previous)
