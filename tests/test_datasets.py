"""Unit tests for :mod:`ideal_functions.datasets`."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ideal_functions.datasets import (Dataset, IdealFunctionDataset, TestDataset,
                                      TrainingDataset)
from ideal_functions.exceptions import (DataLoadError, DataValidationError,
                                        IdealFunctionError)
from ideal_functions.functions import IdealFunction
from tests.helpers import ideal_frame, training_frame, write_csv, write_text


class DatasetLoadingTest(unittest.TestCase):
    """Loading valid and invalid CSV files."""

    def setUp(self):
        """Create a temporary folder for the CSV files of each test."""
        self._folder = tempfile.TemporaryDirectory()
        self.folder = Path(self._folder.name)

    def tearDown(self):
        """Remove the temporary folder."""
        self._folder.cleanup()

    def test_training_dataset_loads_valid_file(self):
        """A valid training file yields a TrainingDataset with four y columns."""
        path = write_csv(training_frame(), self.folder, "train.csv")
        dataset = TrainingDataset.from_csv(path)
        self.assertIsInstance(dataset, TrainingDataset)
        self.assertEqual(dataset.y_columns, ["y1", "y2", "y3", "y4"])
        self.assertEqual(len(dataset), 9)

    def test_ideal_dataset_loads_fifty_functions(self):
        """A valid ideal file yields fifty y columns."""
        path = write_csv(ideal_frame(), self.folder, "ideal.csv")
        dataset = IdealFunctionDataset.from_csv(path)
        self.assertEqual(len(dataset.y_columns), 50)

    def test_missing_file_raises_data_load_error(self):
        """A missing file is reported as DataLoadError."""
        with self.assertRaises(DataLoadError):
            TrainingDataset.from_csv(self.folder / "does_not_exist.csv")

    def test_empty_file_raises_data_load_error(self):
        """An empty file is reported as DataLoadError."""
        path = write_text(self.folder, "empty.csv", "")
        with self.assertRaises(DataLoadError):
            TrainingDataset.from_csv(path)

    def test_user_defined_errors_share_a_base_class(self):
        """All user-defined errors derive from IdealFunctionError."""
        self.assertTrue(issubclass(DataLoadError, IdealFunctionError))
        self.assertTrue(issubclass(DataValidationError, IdealFunctionError))


class DatasetValidationTest(unittest.TestCase):
    """Validation rules of the dataset classes."""

    def test_wrong_number_of_columns_is_rejected(self):
        """Training data with three y columns are rejected."""
        with self.assertRaises(DataValidationError):
            TrainingDataset(training_frame().drop(columns="y4"))

    def test_first_column_must_be_x(self):
        """A first column other than x is rejected."""
        frame = training_frame().rename(columns={"x": "a"})
        with self.assertRaises(DataValidationError):
            TrainingDataset(frame)

    def test_non_numeric_values_are_rejected(self):
        """Non-numeric cells are rejected."""
        frame = training_frame().astype(object)
        frame.loc[2, "y1"] = "abc"
        with self.assertRaises(DataValidationError):
            TrainingDataset(frame)

    def test_missing_values_are_rejected(self):
        """Missing values (NaN) are rejected."""
        frame = training_frame()
        frame.loc[3, "y2"] = float("nan")
        with self.assertRaises(DataValidationError):
            TrainingDataset(frame)

    def test_duplicate_x_rejected_for_functions_but_allowed_for_test_data(self):
        """Repeated x values are only allowed in test data."""
        frame = pd.DataFrame({"x": [1.0, 1.0], "y": [2.0, 3.0]})
        with self.assertRaises(DataValidationError):
            Dataset(frame)
        self.assertEqual(len(TestDataset(frame)), 2)

    def test_column_names_are_normalised_and_rows_sorted(self):
        """Column names are normalised and rows are sorted by x."""
        frame = pd.DataFrame({" X ": [2.0, 1.0], "Y": [4.0, 1.0]})
        dataset = TestDataset(frame)
        self.assertEqual(list(dataset.data.columns), ["x", "y"])
        self.assertEqual(list(dataset.x), [1.0, 2.0])

    def test_data_property_returns_a_copy(self):
        """Changing the returned data does not alter the dataset."""
        dataset = TrainingDataset(training_frame())
        copy = dataset.data
        copy.loc[0, "y1"] = 999.0
        self.assertNotEqual(dataset.data.loc[0, "y1"], 999.0)

    def test_ideal_dataset_creates_numbered_ideal_functions(self):
        """The overridden to_functions returns numbered IdealFunctions."""
        functions = IdealFunctionDataset(ideal_frame()).to_functions()
        self.assertTrue(all(isinstance(f, IdealFunction) for f in functions))
        self.assertEqual([f.number for f in functions], list(range(1, 51)))


class TestDatasetStreamingTest(unittest.TestCase):
    """Line-by-line reading of the test data."""

    def setUp(self):
        """Create a temporary folder for the CSV files of each test."""
        self._folder = tempfile.TemporaryDirectory()
        self.folder = Path(self._folder.name)

    def tearDown(self):
        """Remove the temporary folder."""
        self._folder.cleanup()

    def test_rows_are_yielded_as_float_tuples(self):
        """Each data line is yielded as an (x, y) tuple of floats."""
        path = write_text(self.folder, "test.csv", "x,y\n1.5,2.0\n-3,4.25\n\n")
        self.assertEqual(list(TestDataset.iter_csv(path)), [(1.5, 2.0), (-3.0, 4.25)])

    def test_reading_is_lazy(self):
        """Rows are read one at a time, so earlier rows are available first."""
        path = write_text(self.folder, "test.csv", "x,y\n1,2\nbad,row\n")
        rows = TestDataset.iter_csv(path)
        self.assertEqual(next(rows), (1.0, 2.0))  # First row works before the bad one.
        with self.assertRaises(DataValidationError):
            next(rows)

    def test_wrong_header_is_rejected(self):
        """A header other than x,y is rejected."""
        path = write_text(self.folder, "test.csv", "a,b\n1,2\n")
        with self.assertRaises(DataValidationError):
            list(TestDataset.iter_csv(path))

    def test_wrong_number_of_values_reports_line_number(self):
        """A malformed line is reported with its line number."""
        path = write_text(self.folder, "test.csv", "x,y\n1,2\n3,4,5\n")
        with self.assertRaisesRegex(DataValidationError, "line 3"):
            list(TestDataset.iter_csv(path))

    def test_missing_value_is_rejected(self):
        """A missing value in a test row is rejected."""
        path = write_text(self.folder, "test.csv", "x,y\n1,nan\n")
        with self.assertRaises(DataValidationError):
            list(TestDataset.iter_csv(path))

    def test_missing_file_raises_data_load_error(self):
        """A missing file is reported as DataLoadError."""
        with self.assertRaises(DataLoadError):
            list(TestDataset.iter_csv(self.folder / "missing.csv"))


if __name__ == "__main__":
    unittest.main()
