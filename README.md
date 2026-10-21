# Ideal function mapper

Code for the written assignment of the course DLMDSPWP01 *Programming with Python*
(IU International University of Applied Sciences).

The program

1. loads four training functions (`train.csv`) and fifty ideal functions (`ideal.csv`)
   and stores them in an SQLite database,
2. selects for every training function the ideal function with the smallest sum of
   squared deviations (least squares),
3. reads the test data (`test.csv`) line by line and assigns a point to a chosen ideal
   function if its deviation does not exceed the maximum training deviation times sqrt(2),
4. stores the assigned points in the table `test_mapping` and
5. writes an interactive Bokeh report (`report.html`).

## Project structure

```
ideal_functions/      package with the program logic
    datasets.py       loading and validation of the CSV files
    functions.py      discrete functions and deviation measures
    analysis.py       least-squares selection and test data mapping
    database.py       SQLite persistence via SQLAlchemy
    visualization.py  Bokeh plots
    exceptions.py     user-defined exceptions
tests/                unit and integration tests (unittest)
results/              output of the run on the personal dataset
main.py               command-line entry point
```

## Usage

Python 3.10 or newer is required.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use "source .venv/bin/activate" elsewhere
pip install -r requirements.txt
python main.py --data-dir data --output-dir output --inline
```

The folder given with `--data-dir` must contain `train.csv`, `ideal.csv` and `test.csv`.
The output folder receives `ideal_functions.db` (tables `training_data`,
`ideal_functions`, `test_mapping`) and `report.html`.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

## Results

The folder `results/` contains the console output, the test report, the SQLite database
and the interactive HTML report of the run on the personal dataset.
