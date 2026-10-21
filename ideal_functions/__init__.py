"""Select ideal functions with least squares and map test data to them.

The package is organised in modules that each have one responsibility:

* :mod:`.datasets` - loading and validating the CSV files,
* :mod:`.functions` - discrete functions and deviation measures,
* :mod:`.analysis` - selection of ideal functions and test data mapping,
* :mod:`.database` - persistence in SQLite via SQLAlchemy,
* :mod:`.visualization` - interactive plots with Bokeh,
* :mod:`.exceptions` - user-defined exceptions.
"""

from .analysis import (FunctionMatch, IdealFunctionSelector, MappingResult,
                       TestDataMapper, TestMapping)
from .database import DatabaseManager
from .datasets import Dataset, IdealFunctionDataset, TestDataset, TrainingDataset
from .exceptions import (DatabaseError, DataLoadError, DataValidationError,
                         IdealFunctionError, SelectionError)
from .functions import Function, IdealFunction

__all__ = [
    "DatabaseError", "DatabaseManager", "DataLoadError", "Dataset",
    "DataValidationError", "Function", "FunctionMatch", "IdealFunction",
    "IdealFunctionDataset", "IdealFunctionError", "IdealFunctionSelector",
    "MappingResult", "SelectionError", "TestDataMapper", "TestDataset",
    "TestMapping", "TrainingDataset",
]
