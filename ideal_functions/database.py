"""Persistence of input data and results in a SQLite database via SQLAlchemy."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import Column, Float, Integer, MetaData, Table, create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError

from .analysis import MappingResult
from .datasets import Dataset, IdealFunctionDataset, TrainingDataset
from .exceptions import DatabaseError

TRAINING_TABLE = "training_data"
IDEAL_TABLE = "ideal_functions"
MAPPING_TABLE = "test_mapping"


class DatabaseManager:
    """Create the SQLite database and read or write its tables.

    The class can be used as a context manager, which guarantees that the
    connection pool is released when the ``with`` block is left::

        with DatabaseManager("results.db") as database:
            database.store_training_data(training)
    """

    def __init__(self, path: str | Path | None = None) -> None:
        """Create an engine for a database file or, if ``path`` is ``None``,
        for an in-memory database (useful for unit tests).
        """
        url = "sqlite://" if path is None else f"sqlite:///{Path(path)}"
        self._engine = create_engine(url)
        self._metadata = MetaData()
        self._mapping_table = Table(
            MAPPING_TABLE, self._metadata,
            Column("x", Float, nullable=False),
            Column("y", Float, nullable=False),
            Column("delta_y", Float, nullable=False),
            Column("ideal_function_no", Integer, nullable=False),
        )

    def __enter__(self) -> DatabaseManager:
        """Return the manager itself for use in a ``with`` statement."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Release all connections when the ``with`` block ends."""
        self.close()

    def close(self) -> None:
        """Dispose of the engine and its connection pool."""
        self._engine.dispose()

    def _write_dataset(self, dataset: Dataset, table: str) -> None:
        """Write ``dataset`` to ``table``, replacing an existing table."""
        try:
            dataset.data.to_sql(table, self._engine, if_exists="replace", index=False)
        except (SQLAlchemyError, ValueError) as error:
            raise DatabaseError(f"Could not write table '{table}': {error}") from error

    def store_training_data(self, dataset: TrainingDataset) -> None:
        """Store the training data in a five-column table (x, y1 ... y4)."""
        self._write_dataset(dataset, TRAINING_TABLE)

    def store_ideal_functions(self, dataset: IdealFunctionDataset) -> None:
        """Store the ideal functions in a 51-column table (x, y1 ... y50)."""
        self._write_dataset(dataset, IDEAL_TABLE)

    def store_mapping(self, result: MappingResult) -> None:
        """Store the assigned test points in the four-column result table.

        The table is recreated on every run so that results of an earlier
        run cannot mix with the current ones. It deliberately has no primary
        key: the task prescribes exactly four columns, and the same x-y pair
        may legitimately occur more than once in the test data.
        """
        rows = [{"x": m.x, "y": m.y, "delta_y": m.delta_y,
                 "ideal_function_no": m.ideal_function_no} for m in result.assigned]
        try:
            self._mapping_table.drop(self._engine, checkfirst=True)
            self._mapping_table.create(self._engine)
            if rows:
                with self._engine.begin() as connection:
                    connection.execute(self._mapping_table.insert(), rows)
        except SQLAlchemyError as error:
            raise DatabaseError(f"Could not write table '{MAPPING_TABLE}': {error}") from error

    def table_names(self) -> list[str]:
        """Return the names of all tables in the database."""
        try:
            return inspect(self._engine).get_table_names()
        except SQLAlchemyError as error:
            raise DatabaseError(f"Could not inspect the database: {error}") from error

    def read_table(self, table: str) -> pd.DataFrame:
        """Return the content of ``table`` as a data frame.

        Raises:
            DatabaseError: If the table does not exist or cannot be read.
        """
        if table not in self.table_names():
            raise DatabaseError(f"Table '{table}' does not exist.")
        try:
            with self._engine.connect() as connection:
                return pd.read_sql_table(table, connection)
        except (SQLAlchemyError, ValueError) as error:
            raise DatabaseError(f"Could not read table '{table}': {error}") from error
