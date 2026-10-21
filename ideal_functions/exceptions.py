"""User-defined exceptions of the ``ideal_functions`` package.

All exceptions raised deliberately by the package derive from
:class:`IdealFunctionError`. Calling code can therefore catch every
anticipated, domain-specific failure with a single ``except`` clause while
unexpected programming errors (for example a ``TypeError``) still surface
unchanged.
"""


class IdealFunctionError(Exception):
    """Base class for all user-defined exceptions of this package."""


class DataLoadError(IdealFunctionError):
    """Raised when a CSV file cannot be found, opened or parsed."""


class DataValidationError(IdealFunctionError):
    """Raised when loaded data violate the expected structure or content.

    Examples are a missing ``x`` column, a wrong number of ``y`` columns,
    non-numeric values, missing values or duplicate ``x`` values in a dataset
    that has to describe a function.
    """


class DatabaseError(IdealFunctionError):
    """Raised when reading from or writing to the SQLite database fails."""


class SelectionError(IdealFunctionError):
    """Raised when ideal functions cannot be selected for the training data.

    This happens, for instance, if the training data cover an ``x`` range
    for which the ideal functions are not defined.
    """
