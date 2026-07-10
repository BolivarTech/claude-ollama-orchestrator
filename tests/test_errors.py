# Author: Julian Bolivar
# Version: 1.0.0
# Date: 2026-07-05
"""Verify the domain-exception hierarchy and the deliberate sibling relationship."""

import pytest

from errors import (
    DelegationError,
    InvalidInputError,
    OllamaBackendError,
    OllamaConfigError,
    OllamaPreflightError,
    ValidationError,
)


def test_config_and_preflight_errors_are_validation_error_subclasses():
    assert issubclass(OllamaConfigError, ValidationError)
    assert issubclass(OllamaPreflightError, ValidationError)


def test_backend_and_delegation_errors_are_not_validation_errors():
    assert not issubclass(OllamaBackendError, ValidationError)
    assert not issubclass(DelegationError, ValidationError)


def test_invalid_input_error_is_sibling_not_subclass_of_validation_error():
    # A fail-closed security event must NOT be caught by
    # `except (ValidationError, JSONDecodeError)` in the retry path.
    assert not issubclass(InvalidInputError, ValidationError)
    assert issubclass(InvalidInputError, Exception)


def test_rate_limit_error_is_a_backend_error_subclass():
    # Additive/backward-compatible (R14b/Task 3): MS5 adds this to MS1's hierarchy, but
    # it IS-A OllamaBackendError, so any existing `except OllamaBackendError` (incl. MS1's
    # own tests) still catches it unchanged.
    from errors import RateLimitError

    assert issubclass(RateLimitError, OllamaBackendError)
    with pytest.raises(OllamaBackendError):
        raise RateLimitError("429 exhausted")
