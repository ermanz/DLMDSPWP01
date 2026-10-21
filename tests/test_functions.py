"""Unit tests for :mod:`ideal_functions.functions`."""

import unittest

import numpy as np

from ideal_functions.functions import Function, IdealFunction


class FunctionTest(unittest.TestCase):
    """Evaluation and deviation measures of discrete functions."""

    def setUp(self):
        """Create f(x) = 2x on the points 0, 1, 2, 3 (stored unsorted)."""
        self.function = Function("f", np.array([3.0, 0.0, 2.0, 1.0]),
                                 np.array([6.0, 0.0, 4.0, 2.0]))

    def test_points_are_sorted_by_x(self):
        """Points passed in any order are stored sorted by x."""
        np.testing.assert_array_equal(self.function.x, [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(self.function.y, [0.0, 2.0, 4.0, 6.0])

    def test_evaluation_at_known_point_is_exact(self):
        """Evaluation at a stored point returns the stored value."""
        self.assertEqual(self.function.evaluate(2.0), 4.0)

    def test_evaluation_between_points_interpolates_linearly(self):
        """Evaluation between points interpolates linearly."""
        self.assertAlmostEqual(self.function.evaluate(1.5), 3.0)

    def test_evaluation_of_array_returns_array(self):
        """Array input returns an array of values."""
        np.testing.assert_allclose(self.function.evaluate(np.array([0.5, 2.5])), [1.0, 5.0])

    def test_evaluation_outside_range_is_refused(self):
        """Extrapolation beyond the defined range raises ValueError."""
        with self.assertRaises(ValueError):
            self.function.evaluate(3.5)

    def test_covers(self):
        """covers distinguishes x values inside and outside the range."""
        self.assertTrue(self.function.covers(0.0))
        self.assertFalse(self.function.covers(-0.1))

    def test_deviation_measures(self):
        """Deviations, their sum of squares and maximum are correct."""
        other = Function("g", np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 2.0, 2.0, 6.0]))
        np.testing.assert_allclose(self.function.deviations_from(other), [-1.0, 0.0, 2.0, 0.0])
        self.assertAlmostEqual(self.function.sum_of_squared_deviations(other), 5.0)
        self.assertAlmostEqual(self.function.max_absolute_deviation(other), 2.0)

    def test_invalid_input_is_rejected(self):
        """Arrays of unequal length or empty arrays are rejected."""
        with self.assertRaises(ValueError):
            Function("f", np.array([1.0, 2.0]), np.array([1.0]))
        with self.assertRaises(ValueError):
            Function("f", np.array([]), np.array([]))


class IdealFunctionTest(unittest.TestCase):
    """Behaviour added by the subclass :class:`IdealFunction`."""

    def test_number_is_derived_from_name(self):
        """The number of an ideal function is parsed from its name."""
        function = IdealFunction("y42", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        self.assertEqual(function.number, 42)
        self.assertIsInstance(function, Function)

    def test_invalid_name_is_rejected(self):
        """A name not of the form y<number> is rejected."""
        with self.assertRaises(ValueError):
            IdealFunction("ideal", np.array([0.0]), np.array([0.0]))


if __name__ == "__main__":
    unittest.main()
