"""Loading and validation of the CSV datasets.

The module defines a small class hierarchy. :class:`Dataset` implements
everything that is common to all inputs (reading a CSV file with pandas and
checking that it consists of one ``x`` column and numeric ``y`` columns).
The subclasses only declare what is specific to them, i.e. how many ``y``
columns they must contain and whether ``x`` values may repeat.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from .exceptions import DataLoadError, DataValidationError
from .functions import Function, IdealFunction


class Dataset:
    """A table of x-y pairs with one ``x`` column and one or more ``y`` columns.

    Subclasses customise the validation through two class attributes:

    * ``expected_y_columns`` - required number of ``y`` columns
      (``None`` accepts any positive number).
    * ``requires_unique_x`` - whether every ``x`` value must occur only once,
      which is necessary if the columns are interpreted as functions of ``x``.

    Attributes:
        name: Human-readable name used in log and error messages.
    """

    expected_y_columns: int | None = None
    requires_unique_x: bool = True

    def __init__(self, data: pd.DataFrame, name: str = "dataset") -> None:
        """Validate ``data`` and store a copy sorted by ``x``.

        Args:
            data: Data frame whose first column is named ``x``.
            name: Name of the dataset for messages.

        Raises:
            DataValidationError: If ``data`` does not satisfy the rules of
                the (sub)class.
        """
        self.name = name
        self._data = self._validate(data.copy())

    @classmethod
    def from_csv(cls, path: str | Path, name: str | None = None) -> Dataset:
        """Create a dataset from a CSV file.

        Args:
            path: Location of the CSV file.
            name: Optional dataset name; defaults to the file name.

        Returns:
            An instance of the class on which the method is called, so
            ``TrainingDataset.from_csv(...)`` returns a ``TrainingDataset``.

        Raises:
            DataLoadError: If the file is missing, unreadable or not valid CSV.
            DataValidationError: If the content violates the dataset rules.
        """
        path = Path(path)
        try:
            data = pd.read_csv(path)
        except FileNotFoundError as error:
            raise DataLoadError(f"File not found: {path}") from error
        except pd.errors.EmptyDataError as error:
            raise DataLoadError(f"File is empty: {path}") from error
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as error:
            raise DataLoadError(f"Could not read {path}: {error}") from error
        return cls(data, name=name or path.name)

    def _validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Check structure and content of ``data`` and return it cleaned.

        Column names are normalised to lower case without surrounding
        blanks so that ``X`` and ``x`` are treated alike.
        """
        data.columns = [str(column).strip().lower() for column in data.columns]
        if data.empty:
            raise DataValidationError(f"{self.name}: the dataset contains no rows.")
        if data.columns[0] != "x":
            raise DataValidationError(
                f"{self.name}: the first column must be 'x', found '{data.columns[0]}'."
            )
        y_count = data.shape[1] - 1
        if y_count < 1:
            raise DataValidationError(f"{self.name}: no y column found.")
        if self.expected_y_columns is not None and y_count != self.expected_y_columns:
            raise DataValidationError(
                f"{self.name}: expected {self.expected_y_columns} y columns, found {y_count}."
            )
        try:
            data = data.astype(float)
        except (TypeError, ValueError) as error:
            raise DataValidationError(
                f"{self.name}: all values must be numeric ({error})."
            ) from error
        if data.isna().to_numpy().any():
            raise DataValidationError(f"{self.name}: the dataset contains missing values.")
        if self.requires_unique_x and data["x"].duplicated().any():
            raise DataValidationError(f"{self.name}: x values must be unique.")
        return data.sort_values("x", kind="stable").reset_index(drop=True)

    @property
    def data(self) -> pd.DataFrame:
        """Return a copy of the validated data, so callers cannot corrupt it."""
        return self._data.copy()

    @property
    def x(self) -> np.ndarray:
        """Return the ``x`` values as a NumPy array."""
        return self._data["x"].to_numpy()

    @property
    def y_columns(self) -> list[str]:
        """Return the names of all ``y`` columns in file order."""
        return list(self._data.columns[1:])

    def to_functions(self) -> list[Function]:
        """Interpret every ``y`` column as a discrete function of ``x``."""
        return [Function(column, self.x, self._data[column].to_numpy())
                for column in self.y_columns]

    def __len__(self) -> int:
        """Return the number of rows (x-y pairs) in the dataset."""
        return len(self._data)

    def __repr__(self) -> str:
        """Return a short description useful for debugging."""
        return f"{type(self).__name__}(name={self.name!r}, rows={len(self)})"


class TrainingDataset(Dataset):
    """The four training functions (columns ``x``, ``y1`` ... ``y4``)."""

    expected_y_columns = 4


class IdealFunctionDataset(Dataset):
    """The fifty candidate ideal functions (columns ``x``, ``y1`` ... ``y50``)."""

    expected_y_columns = 50

    def to_functions(self) -> list[IdealFunction]:
        """Return the columns as :class:`IdealFunction` objects.

        Overrides :meth:`Dataset.to_functions` because ideal functions carry
        a number that is needed for the result table.
        """
        return [IdealFunction(column, self.x, self._data[column].to_numpy())
                for column in self.y_columns]


class TestDataset(Dataset):
    """The test data (columns ``x`` and ``y``).

    Test data are a set of observations, not a function, so the same ``x``
    value may occur several times.
    """

    __test__ = False  # Prevent test runners from collecting this class.
    expected_y_columns = 1
    requires_unique_x = False

    @staticmethod
    def iter_csv(path: str | Path) -> Iterator[tuple[float, float]]:
        """Read a test CSV file line by line and yield ``(x, y)`` tuples.

        The generator realises the requirement that the test data are
        processed row by row instead of being loaded into memory at once.

        Args:
            path: Location of the CSV file with the header ``x,y``.

        Yields:
            One ``(x, y)`` tuple per data line.

        Raises:
            DataLoadError: If the file cannot be opened.
            DataValidationError: If the header is wrong or a line does not
                contain exactly two numeric values.
        """
        path = Path(path)
        try:
            handle = path.open(newline="", encoding="utf-8")
        except OSError as error:
            raise DataLoadError(f"Could not open {path}: {error}") from error
        with handle:
            reader = csv.reader(handle)
            header = [column.strip().lower() for column in next(reader, [])]
            if header != ["x", "y"]:
                raise DataValidationError(f"{path.name}: expected header 'x,y', found {header}.")
            for line_number, row in enumerate(reader, start=2):
                if not row:
                    continue  # Tolerate blank lines, e.g. at the end of the file.
                if len(row) != 2:
                    raise DataValidationError(
                        f"{path.name}, line {line_number}: expected 2 values, found {len(row)}."
                    )
                try:
                    x_value, y_value = float(row[0]), float(row[1])
                except ValueError as error:
                    raise DataValidationError(
                        f"{path.name}, line {line_number}: non-numeric value in {row}."
                    ) from error
                if np.isnan(x_value) or np.isnan(y_value):
                    raise DataValidationError(
                        f"{path.name}, line {line_number}: missing value in {row}."
                    )
                yield x_value, y_value
