"""Enumerations used across the pricing service."""
from __future__ import annotations

import enum


class OptionStyle(str, enum.Enum):
    EUROPEAN = "european"
    BARRIER = "barrier"


class OptionType(str, enum.Enum):
    CALL = "call"
    PUT = "put"


class BarrierType(str, enum.Enum):
    UP_AND_IN = "up_and_in"
    UP_AND_OUT = "up_and_out"
    DOWN_AND_IN = "down_and_in"
    DOWN_AND_OUT = "down_and_out"


class ModelType(str, enum.Enum):
    BLACK_SCHOLES = "black_scholes"
    REINER_RUBINSTEIN = "reiner_rubinstein"
