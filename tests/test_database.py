"""Unit tests for :mod:`ideal_functions.database`."""

import tempfile
import unittest
from pathlib import Path

from ideal_functions.analysis import MappingResult, TestMapping
from ideal_functions.database import (IDEAL_TABLE, MAPPING_TABLE, TRAINING_TABLE,
                                      DatabaseManager)
from ideal_functions.datasets import IdealFunctionDataset, TrainingDataset
from ideal_functions.exceptions import DatabaseError
from tests.helpers import ideal_frame, training_frame


class DatabaseManagerTest(unittest.TestCase):
    """Writing and reading the three tables with an in-memory database."""

    def setUp(self):
        """Open a fresh in-memory database for every test."""
        self.database = DatabaseManager()

    def tearDown(self):
        """Release the database connection."""
        self.database.close()

    def test_training_table_has_five_columns(self):
        """The training table has the columns x, y1 ... y4."""
        self.database.store_training_data(TrainingDataset(training_frame()))
        table = self.database.read_table(TRAINING_TABLE)
        self.assertEqual(list(table.columns), ["x", "y1", "y2", "y3", "y4"])
        self.assertEqual(len(table), 9)

    def test_ideal_table_has_fifty_one_columns(self):
        """The ideal function table has 51 columns."""
        self.database.store_ideal_functions(IdealFunctionDataset(ideal_frame()))
        self.assertEqual(self.database.read_table(IDEAL_TABLE).shape, (9, 51))

    def test_mapping_table_has_four_columns_and_stored_values(self):
        """The result table has four columns and keeps duplicate rows."""
        result = MappingResult(assigned=[TestMapping(1.0, 2.0, 0.1, 7),
                                         TestMapping(1.0, 2.0, 0.1, 7)])
        self.database.store_mapping(result)
        table = self.database.read_table(MAPPING_TABLE)
        self.assertEqual(list(table.columns), ["x", "y", "delta_y", "ideal_function_no"])
        self.assertEqual(len(table), 2)  # Duplicate points are kept.
        self.assertEqual(int(table.loc[0, "ideal_function_no"]), 7)

    def test_mapping_table_is_replaced_on_every_run(self):
        """Storing a new result replaces the previous one."""
        self.database.store_mapping(MappingResult(assigned=[TestMapping(1.0, 2.0, 0.1, 7)]))
        self.database.store_mapping(MappingResult())
        self.assertTrue(self.database.read_table(MAPPING_TABLE).empty)

    def test_reading_unknown_table_raises_database_error(self):
        """Reading a non-existent table raises DatabaseError."""
        with self.assertRaises(DatabaseError):
            self.database.read_table("unknown")


class DatabaseFileTest(unittest.TestCase):
    """The database is created as a file and usable as a context manager."""

    def test_database_file_is_created(self):
        """A database file is created and closed by the context manager."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.db"
            with DatabaseManager(path) as database:
                database.store_training_data(TrainingDataset(training_frame()))
                self.assertIn(TRAINING_TABLE, database.table_names())
            self.assertTrue(path.exists())

    def test_invalid_location_raises_database_error(self):
        """An unusable database location raises DatabaseError."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing_folder" / "test.db"
            with DatabaseManager(path) as database:
                with self.assertRaises(DatabaseError):
                    database.store_training_data(TrainingDataset(training_frame()))


if __name__ == "__main__":
    unittest.main()
