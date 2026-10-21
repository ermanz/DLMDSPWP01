"""Shared fixtures for the unit tests: small synthetic CSV files."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

X_VALUES = np.round(np.arange(-2.0, 2.01, 0.5), 1)

#: Training noise added to the four training functions. Its largest absolute
#: value, 0.3, is the maximum deviation every test expects.
NOISE = np.array([0.1, -0.2, 0.3, -0.1, 0.0, 0.2, -0.3, 0.1, -0.1])


def ideal_frame(count: int = 50) -> pd.DataFrame:
    """Return ``count`` ideal functions y_k(x) = k * x + k on ``X_VALUES``.

    The functions are deliberately far apart, so the least-squares choice
    is unambiguous.
    """
    data = {"x": X_VALUES}
    for k in range(1, count + 1):
        data[f"y{k}"] = k * X_VALUES + k
    return pd.DataFrame(data)


def training_frame(chosen: tuple[int, ...] = (3, 7, 12, 40)) -> pd.DataFrame:
    """Return four training functions equal to the ``chosen`` ideal
    functions plus ``NOISE``."""
    ideal = ideal_frame()
    data = {"x": X_VALUES}
    for column, k in enumerate(chosen, start=1):
        data[f"y{column}"] = ideal[f"y{k}"] + NOISE
    return pd.DataFrame(data)


def allowed_deviation() -> float:
    """Return the allowed test deviation for the synthetic data."""
    return 0.3 * math.sqrt(2)


def write_csv(frame: pd.DataFrame, folder: Path, name: str) -> Path:
    """Write ``frame`` to ``folder / name`` and return the path."""
    path = Path(folder) / name
    frame.to_csv(path, index=False)
    return path


def write_text(folder: Path, name: str, text: str) -> Path:
    """Write raw ``text`` to ``folder / name`` and return the path."""
    path = Path(folder) / name
    path.write_text(text, encoding="utf-8")
    return path
