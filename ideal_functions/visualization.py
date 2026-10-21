"""Interactive visualisation of data and results with Bokeh.

Every public method returns a Bokeh layout, so the plots can be inspected
or tested individually; :meth:`Visualizer.save_report` combines them in one
HTML file with a tab per view.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.io import save
from bokeh.layouts import gridplot
from bokeh.models import (ColumnDataSource, DataTable, HoverTool, NumberFormatter,
                          TableColumn, TabPanel, Tabs)
from bokeh.palettes import Category10
from bokeh.plotting import figure
from bokeh.resources import CDN, INLINE

from .analysis import DEFAULT_TOLERANCE_FACTOR, FunctionMatch, MappingResult

UNASSIGNED_COLOR = "#7f7f7f"


class Visualizer:
    """Build Bokeh plots for the training data, the selection and the mapping.

    Attributes:
        matches: The chosen ideal function for every training function.
        result: The outcome of the test data mapping.
        sse_matrix: Least-squares criterion for all function pairs.
        factor: Tolerance factor used for the mapping (``sqrt(2)``).
    """

    def __init__(self, matches: Sequence[FunctionMatch], result: MappingResult,
                 sse_matrix: pd.DataFrame,
                 factor: float = DEFAULT_TOLERANCE_FACTOR,
                 width: int = 520, height: int = 340) -> None:
        """Store the analysis results and the size of a single plot."""
        self.matches = list(matches)
        self.result = result
        self.sse_matrix = sse_matrix
        self.factor = factor
        self._width = width
        self._height = height
        palette = Category10[10]
        self._colors = {m.ideal.number: palette[i % 10] for i, m in enumerate(self.matches)}

    def _figure(self, title: str, **kwargs) -> figure:
        """Create a figure with the common size and tools."""
        return figure(title=title, width=self._width, height=self._height,
                      tools="pan,wheel_zoom,box_zoom,reset,save", **kwargs)

    @staticmethod
    def _legend_above(plot: figure) -> None:
        """Move the legend of ``plot`` above the plot area so it hides no data.

        Clicking a legend entry hides the corresponding glyphs.
        """
        legend = plot.legend[0]
        plot.center.remove(legend)
        legend.orientation = "horizontal"
        legend.click_policy = "hide"
        legend.label_text_font_size = "9pt"
        legend.spacing = 12
        plot.add_layout(legend, "above")

    def _assigned_frame(self, number: int) -> pd.DataFrame:
        """Return the test points assigned to ideal function ``number``."""
        frame = self.result.to_dataframe()
        return frame[frame["ideal_function_no"] == number]

    def _unassigned_frame(self) -> pd.DataFrame:
        """Return the test points that could not be assigned."""
        return pd.DataFrame(self.result.unassigned, columns=["x", "y"])

    def training_fit_plot(self):
        """Plot each training function together with its chosen ideal function."""
        plots = []
        for match in self.matches:
            color = self._colors[match.ideal.number]
            plot = self._figure(f"Training {match.training.name} vs. ideal {match.ideal.name}",
                                x_axis_label="x", y_axis_label="y")
            plot.scatter(match.training.x, match.training.y, size=3, color="black",
                         alpha=0.5, legend_label=f"training {match.training.name}")
            plot.line(match.ideal.x, match.ideal.y, line_width=2, color=color,
                      legend_label=f"ideal {match.ideal.name}")
            self._legend_above(plot)
            plots.append(plot)
        return gridplot(plots, ncols=2)

    def sse_plot(self):
        """Plot the sum of squared deviations of all 50 ideal functions.

        A logarithmic axis is used because the values span many orders of
        magnitude; the selected minimum is highlighted with a large marker.
        """
        plot = self._figure("Least-squares criterion for all ideal functions",
                            x_axis_label="number of ideal function",
                            y_axis_label="sum of squared deviations (log scale)",
                            y_axis_type="log")
        numbers = [int(name[1:]) for name in self.sse_matrix.columns]
        for match in self.matches:
            color = self._colors[match.ideal.number]
            values = self.sse_matrix.loc[match.training.name].to_numpy()
            plot.line(numbers, values, color=color, line_width=1.5,
                      legend_label=f"training {match.training.name}")
            plot.scatter([match.ideal.number], [match.sum_squared_deviation], size=12,
                         color=color, marker="diamond")
        self._legend_above(plot)
        plot.width = 2 * self._width
        return plot

    def mapping_plot(self):
        """Plot the test data together with the chosen ideal functions.

        The first panel shows all test points; in the following panels each
        ideal function is drawn with the test points assigned to it and the
        unassigned points that fall into the visible range.
        """
        unassigned = ColumnDataSource(self._unassigned_frame())
        hover = [("x", "@x{0.0}"), ("y", "@y{0.000}")]
        overview = self._figure("All test data", x_axis_label="x", y_axis_label="y")
        all_points = pd.concat([self.result.to_dataframe()[["x", "y"]], self._unassigned_frame()])
        overview.scatter(all_points["x"], all_points["y"], size=5, color="black")
        plots = [overview]
        for match in self.matches:
            number = match.ideal.number
            color = self._colors[number]
            assigned = ColumnDataSource(self._assigned_frame(number))
            y_min, y_max = match.ideal.y.min(), match.ideal.y.max()
            padding = 0.05 * (y_max - y_min) or 1.0
            plot = self._figure(f"Test points assigned to {match.ideal.name}",
                                x_axis_label="x", y_axis_label="y",
                                y_range=(y_min - padding, y_max + padding))
            plot.line(match.ideal.x, match.ideal.y, color=color, line_width=2,
                      legend_label=f"ideal {match.ideal.name}")
            plot.scatter("x", "y", source=assigned, size=8, color=color,
                         legend_label="assigned")
            plot.scatter("x", "y", source=unassigned, size=8, marker="x",
                         color=UNASSIGNED_COLOR, legend_label="not assigned")
            plot.add_tools(HoverTool(tooltips=hover + [("delta y", "@delta_y{0.000}")]))
            self._legend_above(plot)
            plots.append(plot)
        return gridplot(plots, ncols=2)

    def deviation_plot(self):
        """Plot the deviation of the test points from each ideal function.

        For every chosen function the signed deviation ``y - f(x)`` is shown
        against ``x``. The light band marks the maximum training deviation,
        the darker band the allowed deviation (training maximum times
        ``sqrt(2)``). Points inside the band are assigned; unassigned points
        are drawn in grey, those far outside the band are clipped.
        """
        plots = []
        unassigned = self._unassigned_frame()
        for match in self.matches:
            number = match.ideal.number
            color = self._colors[number]
            allowed = match.allowed_deviation(self.factor)
            limit = 3 * allowed
            x_grid = match.ideal.x
            plot = self._figure(f"Deviation from {match.ideal.name} "
                                f"(allowed |delta y| <= {allowed:.3f})",
                                x_axis_label="x", y_axis_label="y - f(x)",
                                y_range=(-limit, limit))
            plot.varea(x=x_grid, y1=-allowed, y2=allowed, fill_color=color,
                       fill_alpha=0.15, legend_label="allowed band")
            plot.varea(x=x_grid, y1=-match.max_deviation, y2=match.max_deviation,
                       fill_color=color, fill_alpha=0.25, legend_label="training max.")
            assigned = self._assigned_frame(number)
            signed = assigned["y"] - match.ideal.evaluate(assigned["x"].to_numpy())
            plot.scatter(assigned["x"], signed, size=8, color=color, legend_label="assigned")
            inside = unassigned[match.ideal.covers(unassigned["x"].to_numpy())]
            residual = inside["y"] - match.ideal.evaluate(inside["x"].to_numpy())
            edge = 0.97 * limit  # Keep clipped markers fully visible.
            plot.scatter(inside["x"], np.clip(residual, -edge, edge), size=8,
                         marker="x", color=UNASSIGNED_COLOR, legend_label="not assigned")
            self._legend_above(plot)
            plots.append(plot)
        return gridplot(plots, ncols=2)

    def result_table(self) -> DataTable:
        """Return an interactive table of the assigned test points."""
        source = ColumnDataSource(self.result.to_dataframe())
        number = NumberFormatter(format="0.0000")
        columns = [TableColumn(field="x", title="x", formatter=number),
                   TableColumn(field="y", title="y", formatter=number),
                   TableColumn(field="delta_y", title="delta y", formatter=number),
                   TableColumn(field="ideal_function_no", title="no. of ideal function")]
        return DataTable(source=source, columns=columns, width=2 * self._width, height=500)

    def views(self) -> dict[str, object]:
        """Return all views keyed by a short title."""
        return {
            "Training data": self.training_fit_plot(),
            "Least squares": self.sse_plot(),
            "Test data mapping": self.mapping_plot(),
            "Deviations": self.deviation_plot(),
            "Result table": self.result_table(),
        }

    def save_report(self, path: str | Path, inline: bool = False) -> Path:
        """Save all views as tabs of a single stand-alone HTML file.

        Args:
            path: Target file.
            inline: Embed the BokehJS library so that the file works offline.

        Returns:
            The path of the written file.
        """
        path = Path(path)
        tabs = Tabs(tabs=[TabPanel(child=view, title=title)
                          for title, view in self.views().items()])
        save(tabs, filename=path, resources=INLINE if inline else CDN,
             title="Ideal function mapping")
        return path
