"""Command-line entry point of the ideal function mapper.

Usage::

    python main.py --data-dir ../dataset --output-dir output

The program

1. loads ``train.csv`` and ``ideal.csv`` and stores them in SQLite,
2. selects one ideal function per training function (least squares),
3. reads ``test.csv`` line by line and maps every point that satisfies the
   ``sqrt(2)`` criterion, storing the result in the database, and
4. writes an interactive Bokeh report.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ideal_functions import (DatabaseManager, IdealFunctionDataset, IdealFunctionError,
                             IdealFunctionSelector, TestDataMapper, TestDataset,
                             TrainingDataset)
from ideal_functions.visualization import Visualizer

logger = logging.getLogger("ideal_functions")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="folder containing train.csv, ideal.csv and test.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("output"),
                        help="folder for the SQLite database and the HTML report")
    parser.add_argument("--inline", action="store_true",
                        help="embed BokehJS in the report so that it works offline")
    return parser.parse_args(argv)


def run(data_dir: Path, output_dir: Path, inline: bool = False) -> dict:
    """Execute the complete workflow and return a summary of the results.

    Raises:
        IdealFunctionError: If loading, analysing or storing the data fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    training = TrainingDataset.from_csv(data_dir / "train.csv")
    ideal = IdealFunctionDataset.from_csv(data_dir / "ideal.csv")

    with DatabaseManager(output_dir / "ideal_functions.db") as database:
        database.store_training_data(training)
        database.store_ideal_functions(ideal)

        selector = IdealFunctionSelector(training, ideal)
        matches = selector.select()

        mapper = TestDataMapper(matches)
        result = mapper.map_points(TestDataset.iter_csv(data_dir / "test.csv"))
        database.store_mapping(result)
        tables = database.table_names()

    report = Visualizer(matches, result, selector.sum_squared_deviation_matrix())
    report_path = report.save_report(output_dir / "report.html", inline=inline)
    return {"matches": matches, "result": result, "tables": tables, "report": report_path}


def main(argv: list[str] | None = None) -> int:
    """Run the program and translate failures into an exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    arguments = parse_arguments(argv)
    try:
        summary = run(arguments.data_dir, arguments.output_dir, arguments.inline)
    except IdealFunctionError as error:
        logger.error("%s: %s", type(error).__name__, error)
        return 1
    for match in summary["matches"]:
        print(f"{match.training.name} -> {match.ideal.name}: "
              f"SSE = {match.sum_squared_deviation:.4f}, "
              f"max. deviation = {match.max_deviation:.4f}, "
              f"allowed = {match.allowed_deviation():.4f}")
    result = summary["result"]
    print(f"Assigned test points: {len(result.assigned)}, "
          f"not assigned: {len(result.unassigned)}")
    print(f"Tables: {', '.join(summary['tables'])}; report: {summary['report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
