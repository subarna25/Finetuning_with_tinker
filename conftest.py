"""Root-level pytest configuration.

Registers custom markers and provides shared fixtures for the test suite.
"""

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest markers.

    Args:
        config: The pytest configuration object.
    """
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
        " (deselect with '-m not integration')",
    )


@pytest.fixture
def skip_without_tinker_key() -> None:
    """Skip the test if TINKER_API_KEY is not set in the environment.

    Use this fixture in integration tests that require a live Tinker API
    connection.  Unit tests should not use this fixture.

    Example::

        def test_live_training(skip_without_tinker_key):
            ...  # only runs when TINKER_API_KEY is present
    """
    api_key = os.environ.get("TINKER_API_KEY", "").strip()
    if not api_key:
        pytest.skip(
            "TINKER_API_KEY is not set — skipping integration test. "
            "Set the environment variable to run this test."
        )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Automatically skip integration tests when TINKER_API_KEY is absent.

    This hook runs after test collection and adds a skip marker to every
    test decorated with ``@pytest.mark.integration`` when the
    ``TINKER_API_KEY`` environment variable is not set.

    Args:
        config: The pytest configuration object.
        items: The list of collected test items.
    """
    api_key = os.environ.get("TINKER_API_KEY", "").strip()
    if api_key:
        return  # key is present — let integration tests run normally

    skip_marker = pytest.mark.skip(
        reason=(
            "TINKER_API_KEY is not set — skipping integration test. "
            "Set the environment variable to run integration tests."
        )
    )
    for item in items:
        if item.get_closest_marker("integration") is not None:
            item.add_marker(skip_marker)
