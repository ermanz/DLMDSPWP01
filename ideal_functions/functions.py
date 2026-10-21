"""Representation of discrete functions given by x-y pairs."""

from __future__ import annotations

import re

import numpy as np


class Function:
    """A function that is only known at a finite set of points.

    Between the known points the function is evaluated by linear
    interpolation. Evaluating it outside the known ``x`` range is refused,
    because extrapolation would invent data.

    Attributes:
        name: Name of the function, e.g. the CSV column name ``y3``.
    """

    def __init__(self, name: str, x: np.ndarray, y: np.ndarray) -> None:
        """Store the points sorted by ``x``.

        Args:
            name: Name of the function.
            x: The ``x`` values.
            y: The ``y`` values belonging to ``x``.

        Raises:
            ValueError: If ``x`` and ``y`` differ in length or are empty.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape or x.ndim != 1 or x.size == 0:
            raise ValueError("x and y must be non-empty one-dimensional arrays of equal length.")
        order = np.argsort(x, kind="stable")
        self.name = name
        self._x = x[order]
        self._y = y[order]

    @property
    def x(self) -> np.ndarray:
        """Return a copy of the ``x`` values."""
        return self._x.copy()

    @property
    def y(self) -> np.ndarray:
        """Return a copy of the ``y`` values."""
        return self._y.copy()

    def covers(self, x_values: float | np.ndarray) -> np.ndarray | bool:
        """Return whether ``x_values`` lie inside the known ``x`` range."""
        x_values = np.asarray(x_values, dtype=float)
        inside = (x_values >= self._x[0]) & (x_values <= self._x[-1])
        return bool(inside) if inside.ndim == 0 else inside

    def evaluate(self, x_values: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the function by linear interpolation.

        For an ``x`` value that is one of the stored points the stored ``y``
        value is returned exactly.

        Args:
            x_values: A single value or an array of values.

        Returns:
            A float for scalar input, otherwise a NumPy array.

        Raises:
            ValueError: If at least one value lies outside the known range.
        """
        x_array = np.asarray(x_values, dtype=float)
        if not np.all(self.covers(x_array)):
            raise ValueError(
                f"{self.name}: x outside the defined range [{self._x[0]}, {self._x[-1]}]."
            )
        result = np.interp(x_array, self._x, self._y)
        return float(result) if result.ndim == 0 else result

    def deviations_from(self, other: Function) -> np.ndarray:
        """Return ``y - other(x)`` at every point of this function."""
        return self._y - other.evaluate(self._x)

    def sum_of_squared_deviations(self, other: Function) -> float:
        """Return the least-squares criterion between this and ``other``."""
        return float(np.sum(self.deviations_from(other) ** 2))

    def max_absolute_deviation(self, other: Function) -> float:
        """Return the largest absolute deviation between this and ``other``."""
        return float(np.max(np.abs(self.deviations_from(other))))

    def __repr__(self) -> str:
        """Return a short description useful for debugging."""
        return f"{type(self).__name__}(name={self.name!r}, points={self._x.size})"


class IdealFunction(Function):
    """A candidate ideal function, identified by its number (``y1`` ... ``y50``)."""

    _NAME_PATTERN = re.compile(r"^y(\d+)$")

    def __init__(self, name: str, x: np.ndarray, y: np.ndarray) -> None:
        """Create the function and derive its number from ``name``.

        Raises:
            ValueError: If ``name`` does not have the form ``y<number>``.
        """
        match = self._NAME_PATTERN.match(name)
        if match is None:
            raise ValueError(f"Ideal function names must look like 'y42', got {name!r}.")
        super().__init__(name, x, y)
        self.number = int(match.group(1))
