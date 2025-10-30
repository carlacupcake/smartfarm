"""ga_result.py."""
import numpy as np
import plotly.graph_objects as go

from .ga_params import GeneticAlgorithmParams
from .ga_population import Population


class GeneticAlgorithmResult():
    """
        Represents the result of a genetic algorithm run.

        Parameters:
            algo_parameters (GeneticAlgorithmParams)
            final_population (Population)
            lowest_costs (ndarray)
            - Lowest cost values across generations.
            avg_parent_costs (ndarray)
            - Average cost of the top-performing parents across generations.
            optimization_params (OptimizationParams)
        """

    def __init__(
            self,
            ga_params:        GeneticAlgorithmParams,
            final_population: Population,
            lowest_costs:     np.ndarray,
            avg_parent_costs: np.ndarray):

        self.ga_params        = ga_params
        self.final_population = final_population
        self.lowest_costs     = lowest_costs
        self.avg_parent_costs = avg_parent_costs


    def get_table_of_best_designs(self, rows: int = 10):
        """
        Retrieves a table of the top-performing designs from the final population.

        Args:
            rows (int, optional)
            - The number of top designs to retrieve.

        Returns:
            table_of_best_designs (ndarray)
        """

        [unique_values, unique_eff_props, unique_costs] = self.final_population.get_unique_designs()
        table_of_best_designs = np.hstack((unique_values[0:rows, :],
                                           unique_eff_props[0:rows, :],
                                           unique_costs[0:rows].reshape(-1, 1)))
        return table_of_best_designs


    def print_table_of_best_designs(self, rows: int = 10):
        """
        Generates and displays a formatted table of the top-performing designs.

        Args:
            rows (int, optional)

        Returns:
            plotly.graph_objects.Figure
        """

        table_data = self.get_table_of_best_designs(rows)
        headers = [
            "Irrigation Frequency (1/hr)",
            "Irrigation Amount (inches/irrigation)",
            "Fertilizer Frequency (1/hr)",
            "Fertilizer Amount (lb/fertilization)"]

        header_color   = "lavender"
        odd_row_color  = "white"
        even_row_color = "lightgrey"
        if rows % 2 == 0:
            multiplier  = int(rows/2)
            cells_color = [[odd_row_color, even_row_color]*multiplier]
        else:
            multiplier  = int(np.floor(rows/2))
            cells_color = [[odd_row_color, even_row_color]*multiplier]
            cells_color.append(odd_row_color)

        fig = go.Figure(data=[go.Table(
            columnwidth = 1000,
            header = dict(
                values=headers,
                fill_color=header_color,
                align="left",
                font=dict(size=12),
                height=30
            ),
            cells = dict(
                values=[table_data[:, i] for i in range(table_data.shape[1])],
                fill_color=cells_color,
                align="left",
                font=dict(size=12),
                height=30,
            )
        )])

        # Update layout for horizontal scrolling
        fig.update_layout(
            title="Optimal Properties Recommended by Genetic Algorithm",
            title_font_size=20,
            title_x=0.2,
            margin=dict(l=0, r=0, t=40, b=0),
            height=400,
            autosize=True
        )

        return fig


    def plot_optimization_results(self):
        """
        Generates a plot visualizing the optimization convergence over generations.

        Returns:
            plotly.graph_objects.Figure
        """

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(self.ga_params.num_generations)),
            y=self.avg_parent_costs,
            mode="lines",
            name="Avg. of top 10 performers"
        ))

        fig.add_trace(go.Scatter(
            x=list(range(self.ga_params.num_generations)),
            y=self.lowest_costs,
            mode="lines",
            name="Best costs"
        ))

        fig.update_layout(
            title="Convergence of Genetic Algorithm",
            title_x=0.25,
            xaxis_title="Generation",
            yaxis_title="Cost",
            legend=dict(
                font=dict(size=14),
                x=1,
                y=1,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.5)"
            ),
            title_font_size=24,
            xaxis_title_font_size=20,
            yaxis_title_font_size=20,
            width=600,
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        return fig
