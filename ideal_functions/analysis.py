"""Selection of ideal functions and mapping of test data.

The module contains the two analytical steps of the program:

1. :class:`IdealFunctionSelector` chooses, for every training function, the
   ideal function with the smallest sum of squared deviations
   (least-squares criterion).
2. :class:`TestDataMapper` assigns a test point to one of the chosen ideal
   functions if its deviation does not exceed the largest training deviation
   of that function multiplied by ``sqrt(2)``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import pandas as pd

from .datasets import IdealFunctionDataset, TrainingDataset
from .exceptions import SelectionError
from .functions import Function, IdealFunction

logger = logging.getLogger(__name__)

#: Factor by which a test deviation may exceed the maximum training deviation.
DEFAULT_TOLERANCE_FACTOR = math.sqrt(2)


@dataclass(frozen=True)
class FunctionMatch:
    """The ideal function chosen for one training function.

    Attributes:
        training: The training function.
        ideal: The ideal function with the smallest sum of squared deviations.
        sum_squared_deviation: Value of the least-squares criterion.
        max_deviation: Largest absolute deviation between both functions.
    """

    training: Function
    ideal: IdealFunction
    sum_squared_deviation: float
    max_deviation: float

    def allowed_deviation(self, factor: float = DEFAULT_TOLERANCE_FACTOR) -> float:
        """Return the largest deviation a test point may have to be assigned."""
        return self.max_deviation * factor


@dataclass(frozen=True)
class TestMapping:
    """A test point that was assigned to an ideal function.

    Attributes:
        x: x value of the test point.
        y: y value of the test point.
        delta_y: Absolute deviation between test point and ideal function.
        ideal_function_no: Number of the ideal function (e.g. 42 for ``y42``).
    """

    __test__ = False  # Prevent test runners from collecting this class.
    x: float
    y: float
    delta_y: float
    ideal_function_no: int


@dataclass
class MappingResult:
    """Outcome of mapping a whole test dataset.

    Attributes:
        assigned: Test points that satisfy the criterion for some function.
        unassigned: ``(x, y)`` tuples of the points that could not be assigned.
    """

    assigned: list[TestMapping] = field(default_factory=list)
    unassigned: list[tuple[float, float]] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the assigned points in the layout of the result table."""
        columns = ["x", "y", "delta_y", "ideal_function_no"]
        rows = [(m.x, m.y, m.delta_y, m.ideal_function_no) for m in self.assigned]
        return pd.DataFrame(rows, columns=columns).astype({"ideal_function_no": "int64"})


class IdealFunctionSelector:
    """Choose the best-fitting ideal function for every training function."""

    def __init__(self, training: TrainingDataset, ideal: IdealFunctionDataset) -> None:
        """Convert both datasets into function objects.

        Raises:
            SelectionError: If the training data contain ``x`` values for
                which the ideal functions are not defined.
        """
        self._training_functions = training.to_functions()
        self._ideal_functions = ideal.to_functions()
        reference = self._ideal_functions[0]
        if not all(reference.covers(training.x)):
            raise SelectionError(
                "The training data contain x values outside the range of the ideal functions."
            )

    def sum_squared_deviation_matrix(self) -> pd.DataFrame:
        """Return the least-squares criterion for every pair of functions.

        Returns:
            Data frame with one row per training function and one column per
            ideal function.
        """
        matrix = {
            ideal.name: [training.sum_of_squared_deviations(ideal)
                         for training in self._training_functions]
            for ideal in self._ideal_functions
        }
        index = [training.name for training in self._training_functions]
        return pd.DataFrame(matrix, index=index)

    def select(self) -> list[FunctionMatch]:
        """Return one :class:`FunctionMatch` per training function.

        If two ideal functions reach exactly the same minimum, the one with
        the lower number is taken, which keeps the result deterministic.
        """
        matrix = self.sum_squared_deviation_matrix()
        ideal_by_name = {ideal.name: ideal for ideal in self._ideal_functions}
        matches = []
        for training in self._training_functions:
            best_name = matrix.loc[training.name].idxmin()
            ideal = ideal_by_name[best_name]
            match = FunctionMatch(
                training=training,
                ideal=ideal,
                sum_squared_deviation=float(matrix.loc[training.name, best_name]),
                max_deviation=training.max_absolute_deviation(ideal),
            )
            logger.info("Training function %s -> ideal function %s (SSE = %.4f)",
                        training.name, ideal.name, match.sum_squared_deviation)
            matches.append(match)
        chosen = [match.ideal.number for match in matches]
        if len(set(chosen)) < len(chosen):
            logger.warning("Several training functions share an ideal function: %s", chosen)
        return matches


class TestDataMapper:
    """Assign test points to the chosen ideal functions."""

    __test__ = False  # Prevent test runners from collecting this class.

    def __init__(self, matches: Sequence[FunctionMatch],
                 tolerance_factor: float = DEFAULT_TOLERANCE_FACTOR) -> None:
        """Store the chosen functions and the tolerance factor.

        Raises:
            ValueError: If no match is given or the factor is not positive.
        """
        if not matches:
            raise ValueError("At least one chosen ideal function is required.")
        if tolerance_factor <= 0:
            raise ValueError("The tolerance factor must be positive.")
        self._matches = list(matches)
        self._factor = tolerance_factor

    def map_point(self, x: float, y: float) -> TestMapping | None:
        """Assign a single test point or return ``None``.

        A function qualifies if the point lies in its ``x`` range and its
        absolute deviation does not exceed the allowed deviation; a
        relative tolerance absorbs floating-point rounding at the boundary.
        If several functions qualify, the one with the smallest deviation
        relative to its allowed deviation is chosen.
        """
        best: tuple[float, TestMapping] | None = None
        for match in self._matches:
            if not match.ideal.covers(x):
                continue
            deviation = abs(y - match.ideal.evaluate(x))
            allowed = match.allowed_deviation(self._factor)
            if deviation > allowed and not math.isclose(deviation, allowed, rel_tol=1e-9):
                continue
            relative = deviation / allowed if allowed > 0 else 0.0
            if best is None or relative < best[0]:
                best = (relative, TestMapping(x, y, deviation, match.ideal.number))
        return None if best is None else best[1]

    def map_points(self, points: Iterable[tuple[float, float]]) -> MappingResult:
        """Map every ``(x, y)`` tuple delivered by ``points``.

        ``points`` may be a generator, so the test data can be streamed line
        by line from the CSV file.
        """
        result = MappingResult()
        for x, y in points:
            mapping = self.map_point(x, y)
            if mapping is None:
                result.unassigned.append((x, y))
            else:
                result.assigned.append(mapping)
        logger.info("Assigned %d test points, %d remain unassigned.",
                    len(result.assigned), len(result.unassigned))
        return result
