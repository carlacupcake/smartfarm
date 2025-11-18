"""ga_result.py."""
import numpy as np
import matplotlib.pyplot as plt

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

        [unique_values, unique_costs] = self.final_population.get_unique_designs()
        table_of_best_designs = np.hstack((unique_values[0:rows, :],
                                           unique_costs[0:rows].reshape(-1, 1)))
        return table_of_best_designs


    def print_table_of_best_designs(self, rows: int = 10):
        """
        Generates and displays a formatted table of the top-performing designs
        using matplotlib.

        Args:
            rows (int, optional): Number of designs to include.

        Returns:
            matplotlib.figure.Figure
        """

        # Get data: expected shape (rows, 4)
        table_data = self.get_table_of_best_designs(rows)

        headers = [
            "Irrigation Frequency (1/hr)",
            "Irrigation Amount (inches/irrigation)",
            "Fertilizer Frequency (1/hr)",
            "Fertilizer Amount (lb/fertilization)",
        ]

        # Create figure and axis
        # Height scales a bit with number of rows so it doesn't get cramped
        fig, ax = plt.subplots(figsize=(10, 0.4 * rows + 1))

        ax.axis("off")
        ax.axis("tight")

        # Matplotlib's table expects rows, not columns
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            loc="center",
        )

        # Some basic formatting
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(headers))))

        ax.set_title(
            "Optimal Properties Recommended by Genetic Algorithm",
            pad=20,
            fontsize=14,
        )

        fig.tight_layout()
        return fig


    def plot_optimization_results(self):
        """
        Generates a plot visualizing optimization convergence over generations
        using matplotlib.

        Returns:
            matplotlib.figure.Figure
        """

        gens = list(range(self.ga_params.num_generations))

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(gens, self.avg_parent_costs, label="Avg. of top 10 performers")
        ax.plot(gens, self.lowest_costs, label="Best costs")

        ax.set_title("Convergence of Genetic Algorithm", fontsize=16, pad=15)
        ax.set_xlabel("Generation", fontsize=13)
        ax.set_ylabel("Cost", fontsize=13)

        ax.legend(
            fontsize=12,
            loc="upper right",
            framealpha=0.5
        )

        fig.tight_layout()
        return
