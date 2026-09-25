"""Distinguishable service-level error types.

Numerical library exceptions must never leak through the API; they are
translated into one of the error codes defined here.
"""
from __future__ import annotations

import enum


class ErrorCode(str, enum.Enum):
    INVALID_PARAMETER = "INVALID_PARAMETER"
    CONTRACT_EXPIRED = "CONTRACT_EXPIRED"
    MODEL_NOT_APPLICABLE = "MODEL_NOT_APPLICABLE"
    INVALID_CONTRACT_SPEC = "INVALID_CONTRACT_SPEC"
    NO_QUOTES = "NO_QUOTES"
    CALIBRATION_NO_SOLUTION = "CALIBRATION_NO_SOLUTION"
    CALIBRATION_DID_NOT_CONVERGE = "CALIBRATION_DID_NOT_CONVERGE"
    NUMERICAL_INSTABILITY = "NUMERICAL_INSTABILITY"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_STALE = "SESSION_STALE"
    EMPTY_BATCH = "EMPTY_BATCH"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class PricingError(Exception):
    def __init__(self, code: ErrorCode, message: str, detail: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}
