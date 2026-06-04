"""Unit tests for qwen3_rl_pipeline/clients.py.

All tests mock the ``tinker`` module via ``sys.modules`` injection so that the
test suite runs correctly even when the ``tinker`` package is not installed.
"""

from __future__ import annotations

import logging
import sys
import types
import unittest.mock as mock

import pytest

from qwen3_rl_pipeline.clients import TinkerClients, initialize_clients


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_tinker() -> types.ModuleType:
    """Return a minimal fake ``tinker`` module suitable for patching."""
    mock_tinker = types.ModuleType("tinker")

    mock_service_client_instance = mock.MagicMock(name="service_client_instance")
    mock_training_client = mock.MagicMock(name="training_client")
    mock_sampling_client = mock.MagicMock(name="sampling_client")

    mock_service_client_instance.create_lora_training_client.return_value = (
        mock_training_client
    )
    mock_service_client_instance.create_sampling_client.return_value = (
        mock_sampling_client
    )

    mock_service_client_cls = mock.MagicMock(
        name="ServiceClient",
        return_value=mock_service_client_instance,
    )
    mock_tinker.ServiceClient = mock_service_client_cls  # type: ignore[attr-defined]

    return mock_tinker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMissingApiKey:
    """EnvironmentError is raised before any SDK call when the key is absent."""

    def test_missing_api_key_raises_before_sdk_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unset TINKER_API_KEY must raise EnvironmentError; ServiceClient never called.

        Validates: Requirements 1.4, 1.5
        """
        monkeypatch.delenv("TINKER_API_KEY", raising=False)

        mock_tinker = _make_mock_tinker()
        with mock.patch.dict(sys.modules, {"tinker": mock_tinker}):
            with pytest.raises(EnvironmentError, match="TINKER_API_KEY"):
                initialize_clients()

            mock_tinker.ServiceClient.assert_not_called()

    def test_empty_api_key_raises_before_sdk_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty TINKER_API_KEY (whitespace-only) must raise EnvironmentError.

        Validates: Requirements 1.4, 1.5
        """
        monkeypatch.setenv("TINKER_API_KEY", "   ")

        mock_tinker = _make_mock_tinker()
        with mock.patch.dict(sys.modules, {"tinker": mock_tinker}):
            with pytest.raises(EnvironmentError, match="TINKER_API_KEY"):
                initialize_clients()

            mock_tinker.ServiceClient.assert_not_called()


class TestSdkException:
    """SDK exceptions are logged at ERROR and re-raised."""

    def test_sdk_exception_is_logged_and_reraised(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A RuntimeError from ServiceClient must be logged at ERROR and re-raised.

        Validates: Requirements 3.4
        """
        monkeypatch.setenv("TINKER_API_KEY", "valid-key")

        mock_tinker = types.ModuleType("tinker")
        mock_tinker.ServiceClient = mock.MagicMock(  # type: ignore[attr-defined]
            side_effect=RuntimeError("sdk error")
        )

        with mock.patch.dict(sys.modules, {"tinker": mock_tinker}):
            with caplog.at_level(logging.ERROR, logger="qwen3_rl_pipeline"):
                with pytest.raises(RuntimeError, match="sdk error"):
                    initialize_clients()

        assert any(
            record.levelno == logging.ERROR
            for record in caplog.records
        ), "Expected at least one ERROR-level log record"


class TestSuccessfulInitialization:
    """All three clients are constructed in the correct order."""

    def test_all_three_clients_constructed_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ServiceClient, create_lora_training_client, and create_sampling_client
        must all be called with the correct arguments, and the returned
        TinkerClients must hold the right objects.

        Validates: Requirements 3.1, 3.2, 3.3
        """
        monkeypatch.setenv("TINKER_API_KEY", "valid-key")

        mock_tinker = _make_mock_tinker()
        service_instance = mock_tinker.ServiceClient.return_value
        expected_training = service_instance.create_lora_training_client.return_value
        expected_sampling = service_instance.create_sampling_client.return_value

        with mock.patch.dict(sys.modules, {"tinker": mock_tinker}):
            result = initialize_clients(base_model="Qwen/Qwen3-8B", rank=32)

        # ServiceClient() called once with no arguments
        mock_tinker.ServiceClient.assert_called_once_with()

        # Training client created with correct kwargs
        service_instance.create_lora_training_client.assert_called_once_with(
            base_model="Qwen/Qwen3-8B",
            rank=32,
        )

        # Sampling client created with correct kwargs
        service_instance.create_sampling_client.assert_called_once_with(
            base_model="Qwen/Qwen3-8B",
        )

        # Returned dataclass holds the right objects
        assert isinstance(result, TinkerClients)
        assert result.service_client is service_instance
        assert result.training_client is expected_training
        assert result.sampling_client is expected_sampling

        # Verify call order: training client before sampling client
        call_order = service_instance.mock_calls
        method_names = [c[0] for c in call_order]
        assert method_names.index("create_lora_training_client") < method_names.index(
            "create_sampling_client"
        ), "create_lora_training_client must be called before create_sampling_client"
