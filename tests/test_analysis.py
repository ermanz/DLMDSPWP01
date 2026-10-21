"""Unit tests for :mod:`ideal_functions.analysis`."""

import math
import unittest

import numpy as np
import pandas as pd

from ideal_functions.analysis import (FunctionMatch, IdealFunctionSelector, MappingResult,
                                      TestDataMapper, TestMapping)
from ideal_functions.datasets import IdealFunctionDataset, TrainingDataset
from ideal_functions.exceptions import SelectionError
from ideal_functions.functions import Function, IdealFunction
from tests.helpers import allowed_deviation, ideal_frame, training_frame


class IdealFunctionSelectorTest(unittest.TestCase):
    """Least-squares selection on synthetic data with a known answer."""

    def setUp(self):
        """Build a selector for training data derived from y3, y7, y12, y40."""
        self.selector = IdealFunctionSelector(TrainingDataset(training_frame()),
                                              IdealFunctionDataset(ideal_frame()))

    def test_matrix_has_one_row_per_training_and_column_per_ideal_function(self):
        """The criterion matrix has shape 4 x 50."""
        matrix = self.selector.sum_squared_deviation_matrix()
        self.assertEqual(matrix.shape, (4, 50))

    def test_selects_the_generating_functions(self):
        """The functions used to generate the training data are selected."""
        numbers = [match.ideal.number for match in self.selector.select()]
        self.assertEqual(numbers, [3, 7, 12, 40])

    def test_criterion_and_maximum_deviation(self):
        """Criterion, maximum deviation and tolerance match hand calculation."""
        match = self.selector.select()[0]
        self.assertAlmostEqual(match.sum_squared_deviation, 0.30)  # Sum of NOISE squared.
        self.assertAlmostEqual(match.max_deviation, 0.3)
        self.assertAlmostEqual(match.allowed_deviation(), 0.3 * math.sqrt(2))

    def test_training_range_outside_ideal_range_is_rejected(self):
        """Training x values outside the ideal range raise SelectionError."""
        training = training_frame()
        training["x"] = training["x"] + 10
        with self.assertRaises(SelectionError):
            IdealFunctionSelector(TrainingDataset(training), IdealFunctionDataset(ideal_frame()))


class TestDataMapperTest(unittest.TestCase):
    """Mapping rule including boundary cases."""

    def setUp(self):
        """Create two ideal functions, y1(x) = 0 and y2(x) = 1, each with
        a maximum training deviation of 0.3."""
        x = np.linspace(-2.0, 2.0, 5)
        self.flat = IdealFunction("y1", x, np.zeros_like(x))
        self.one = IdealFunction("y2", x, np.ones_like(x))
        training = Function("t", x, np.zeros_like(x))
        self.matches = [FunctionMatch(training, self.flat, 0.0, 0.3),
                        FunctionMatch(training, self.one, 0.0, 0.3)]
        self.mapper = TestDataMapper(self.matches)

    def test_point_within_tolerance_is_assigned(self):
        """A point within the tolerance is assigned with its deviation."""
        mapping = self.mapper.map_point(0.0, 0.2)
        self.assertEqual(mapping, TestMapping(0.0, 0.2, 0.2, 1))

    def test_point_exactly_on_boundary_is_assigned(self):
        """A point exactly on the limit sqrt(2) * D is assigned."""
        mapping = self.mapper.map_point(1.0, allowed_deviation())
        self.assertIsNotNone(mapping)
        self.assertEqual(mapping.ideal_function_no, 1)

    def test_point_beyond_tolerance_is_not_assigned(self):
        """A point beyond the tolerance of every function is not assigned."""
        self.assertIsNone(self.mapper.map_point(0.0, 0.5))

    def test_closest_function_wins_if_several_qualify(self):
        """If two functions qualify, the closer one is chosen."""
        training = self.matches[0].training
        mapper = TestDataMapper([FunctionMatch(training, self.flat, 0.0, 0.5),
                                 FunctionMatch(training, self.one, 0.0, 0.5)])
        mapping = mapper.map_point(0.0, 0.6)  # 0.6 from y1, 0.4 from y2, both allowed.
        self.assertEqual(mapping.ideal_function_no, 2)
        self.assertAlmostEqual(mapping.delta_y, 0.4)

    def test_tie_break_uses_deviation_relative_to_tolerance(self):
        """Several candidates are compared relative to their own tolerance."""
        x = np.linspace(-2.0, 2.0, 5)
        narrow = IdealFunction("y5", x, np.zeros_like(x))
        wide = IdealFunction("y6", x, np.full_like(x, 0.42))
        training = self.matches[0].training
        mapper = TestDataMapper([FunctionMatch(training, narrow, 0.0, 0.1),
                                 FunctionMatch(training, wide, 0.0, 1.0)])
        # 0.12 from y5 uses 85 % of its tolerance, 0.30 from y6 only 21 %.
        mapping = mapper.map_point(0.0, 0.12)
        self.assertEqual(mapping.ideal_function_no, 6)

    def test_point_outside_x_range_is_not_assigned(self):
        """A point outside the x range of all functions is not assigned."""
        self.assertIsNone(self.mapper.map_point(5.0, 0.0))

    def test_map_points_accepts_generator_and_separates_results(self):
        """A generator of points is mapped into assigned and unassigned points."""
        points = (point for point in [(0.0, 0.1), (0.0, 0.5), (1.0, 1.1)])
        result = self.mapper.map_points(points)
        self.assertEqual(len(result.assigned), 2)
        self.assertEqual(result.unassigned, [(0.0, 0.5)])

    def test_invalid_configuration_is_rejected(self):
        """An empty list of matches or a non-positive factor is rejected."""
        with self.assertRaises(ValueError):
            TestDataMapper([])
        with self.assertRaises(ValueError):
            TestDataMapper(self.matches, tolerance_factor=0)


class MappingResultTest(unittest.TestCase):
    """Conversion of the mapping result into the table layout."""

    def test_dataframe_has_the_four_required_columns(self):
        """The result frame has the four columns of the result table."""
        result = MappingResult(assigned=[TestMapping(1.0, 2.0, 0.1, 7)])
        frame = result.to_dataframe()
        self.assertEqual(list(frame.columns), ["x", "y", "delta_y", "ideal_function_no"])
        self.assertEqual(frame.loc[0, "ideal_function_no"], 7)

    def test_empty_result_gives_empty_dataframe(self):
        """An empty result converts into an empty data frame."""
        frame = MappingResult().to_dataframe()
        self.assertTrue(frame.empty)
        self.assertIsInstance(frame, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
