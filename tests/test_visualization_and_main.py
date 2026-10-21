"""Tests for :mod:`ideal_functions.visualization` and the program entry point."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from bokeh.models import DataTable, GridPlot
from bokeh.plotting import figure

import main
from ideal_functions.analysis import IdealFunctionSelector, TestDataMapper
from ideal_functions.database import MAPPING_TABLE, DatabaseManager
from ideal_functions.datasets import IdealFunctionDataset, TrainingDataset
from ideal_functions.visualization import Visualizer
from tests.helpers import ideal_frame, training_frame, write_csv, write_text


class VisualizerTest(unittest.TestCase):
    """The visualizer builds every view and writes an HTML report."""

    def setUp(self):
        """Run the analysis on synthetic data and create a visualizer."""
        selector = IdealFunctionSelector(TrainingDataset(training_frame()),
                                         IdealFunctionDataset(ideal_frame()))
        with self.assertLogs("ideal_functions", level="INFO"):
            matches = selector.select()
            # y3(x) = 3x + 3, so (0, 3.1) is assigned and (0, 50) is not.
            result = TestDataMapper(matches).map_points([(0.0, 3.1), (0.0, 50.0)])
        self.visualizer = Visualizer(matches, result, selector.sum_squared_deviation_matrix())

    def test_all_views_are_created_with_expected_types(self):
        """All five views are created with the expected Bokeh types."""
        views = self.visualizer.views()
        self.assertEqual(len(views), 5)
        self.assertIsInstance(views["Least squares"], figure)
        self.assertIsInstance(views["Deviations"], GridPlot)
        self.assertIsInstance(views["Result table"], DataTable)

    def test_grid_plots_contain_one_panel_per_function(self):
        """The grid plots contain one panel per chosen function."""
        self.assertEqual(len(self.visualizer.training_fit_plot().children), 4)
        self.assertEqual(len(self.visualizer.mapping_plot().children), 5)  # Overview + 4.

    def test_report_is_written(self):
        """The HTML report is written to disk."""
        with tempfile.TemporaryDirectory() as folder:
            path = self.visualizer.save_report(Path(folder) / "report.html")
            self.assertTrue(path.exists())
            self.assertIn("Bokeh", path.read_text(encoding="utf-8"))


class MainIntegrationTest(unittest.TestCase):
    """End-to-end run of the program on synthetic CSV files."""

    def setUp(self):
        """Write the three input files into a temporary folder."""
        self._folder = tempfile.TemporaryDirectory()
        self.folder = Path(self._folder.name)
        write_csv(training_frame(), self.folder, "train.csv")
        write_csv(ideal_frame(), self.folder, "ideal.csv")
        # The first point lies on y3, the second matches no chosen function.
        write_text(self.folder, "test.csv", "x,y\n1.0,6.1\n1.0,-100\n")

    def tearDown(self):
        """Remove the temporary folder."""
        self._folder.cleanup()

    def test_successful_run_creates_database_and_report(self):
        """An end-to-end run stores the expected mapping and writes the report."""
        output = self.folder / "out"
        with redirect_stdout(io.StringIO()), self.assertLogs("ideal_functions", level="INFO"):
            exit_code = main.main(["--data-dir", str(self.folder), "--output-dir", str(output)])
        self.assertEqual(exit_code, 0)
        self.assertTrue((output / "report.html").exists())
        with DatabaseManager(output / "ideal_functions.db") as database:
            table = database.read_table(MAPPING_TABLE)
        self.assertEqual(len(table), 1)
        self.assertEqual(int(table.loc[0, "ideal_function_no"]), 3)
        self.assertTrue(np.isclose(table.loc[0, "delta_y"], 0.1))

    def test_missing_input_returns_error_code(self):
        """A missing input file leads to exit code 1 and an error message."""
        (self.folder / "ideal.csv").unlink()
        with self.assertLogs("ideal_functions", level="ERROR"):
            exit_code = main.main(["--data-dir", str(self.folder),
                                   "--output-dir", str(self.folder / "out")])
        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
