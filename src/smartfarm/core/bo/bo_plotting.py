# bo_plotting.py
"""
Plotting utilities for Bayesian Optimization results.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple

from .bo import BOResult


class BOPlotting:
    """
    Plotting class for Bayesian Optimization results.

    Uses matplotlib with LaTeX rendering and color-blind friendly palettes.
    Requires PlottingParams and PlottingColors to be applied before use.
    """

    def __init__(self, result: BOResult, colors=None):
        """
        Initialize BOPlotting.

        Args:
            result: BOResult object containing optimization results
            colors: PlottingColors object (optional, for consistent coloring)
        """
        self.result = result
        self.study = result.study
        self.colors = colors

    def _get_color(self, name: str) -> str:
        """Get a color, using PlottingColors if available."""
        if self.colors is None:
            # Fallback colors
            defaults = {
                'primary': '#007D34',
                'secondary': '#00538A',
                'accent': '#C10020',
                'neutral': '#817066',
                'success': '#007D34',
                'warning': '#FF6800',
                'purple': '#803E75',
            }
            return defaults.get(name, '#007D34')

        color_map = {
            'primary': self.colors.vivid_green,
            'secondary': self.colors.strong_blue,
            'accent': self.colors.vivid_red,
            'neutral': self.colors.medium_gray,
            'success': self.colors.vivid_green,
            'warning': self.colors.vivid_orange,
            'purple': self.colors.strong_purple,
        }
        return color_map.get(name, self.colors.vivid_green)

    def plot_optimization_history(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (10, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot optimization history showing trial values and best-so-far.

        Args:
            ax: Optional matplotlib axes
            figsize: Figure size if creating new figure
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Get trial data
        trials_sorted = sorted(self.study.trials, key=lambda t: t.number)
        trial_numbers = [t.number for t in trials_sorted]
        trial_values = [t.value for t in trials_sorted]

        # Compute best-so-far
        if self.result.bo_params.direction == "maximize":
            best_values = np.maximum.accumulate(trial_values)
        else:
            best_values = np.minimum.accumulate(trial_values)

        # Plot
        ax.scatter(trial_numbers, trial_values, alpha=0.6, s=40,
                   color=self._get_color('secondary'), label='Trial Value')
        ax.plot(trial_numbers, best_values, linewidth=2,
                color=self._get_color('accent'), label='Best So Far')

        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Objective Value')
        ax.set_title('Bayesian Optimization History')
        ax.legend(loc='lower right' if self.result.bo_params.direction == "maximize"
                  else 'upper right')

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_param_importances(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot parameter importances using fANOVA.

        Args:
            ax: Optional matplotlib axes
            figsize: Figure size if creating new figure
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Get importances
        from optuna.importance import get_param_importances
        importances = get_param_importances(self.study)

        # Sort by importance
        param_names = list(importances.keys())
        importance_values = list(importances.values())
        sorted_idx = np.argsort(importance_values)
        param_names_sorted = [param_names[i] for i in sorted_idx]
        importance_sorted = [importance_values[i] for i in sorted_idx]

        # Plot
        bars = ax.barh(param_names_sorted, importance_sorted,
                       color=self._get_color('secondary'), alpha=0.85)

        ax.set_xlabel('Importance (fANOVA)')
        ax.set_title('Parameter Importances')

        # Add value labels
        for bar, val in zip(bars, importance_sorted):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=10)

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_parallel_coordinates(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (14, 6),
        top_percentile: float = 0.8,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot parallel coordinates showing parameter configurations.

        Args:
            ax: Optional matplotlib axes
            figsize: Figure size if creating new figure
            top_percentile: Percentile threshold for highlighting good trials
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Get trials dataframe
        trials_df = self.result.get_trials_dataframe()
        param_cols = [col for col in trials_df.columns if col.startswith('params_')]
        plot_df = trials_df[['value'] + param_cols].copy()

        # Normalize parameters to [0, 1]
        for col in param_cols:
            col_min, col_max = plot_df[col].min(), plot_df[col].max()
            if col_max > col_min:
                plot_df[col] = (plot_df[col] - col_min) / (col_max - col_min)

        # Plot each trial
        value_range = plot_df['value'].max() - plot_df['value'].min()
        threshold = plot_df['value'].quantile(top_percentile)

        for _, row in plot_df.iterrows():
            values = row[param_cols].values
            alpha = 0.3 + 0.7 * (row['value'] - plot_df['value'].min()) / value_range
            is_top = row['value'] >= threshold
            color = self._get_color('success') if is_top else self._get_color('neutral')
            ax.plot(range(len(param_cols)), values, alpha=alpha, linewidth=1, color=color)

        # Highlight best trial
        best_row = plot_df.loc[plot_df['value'].idxmax()]
        ax.plot(range(len(param_cols)), best_row[param_cols].values,
                linewidth=3, color=self._get_color('accent'), label='Best Trial')

        ax.set_xticks(range(len(param_cols)))
        ax.set_xticklabels([col.replace('params_', '') for col in param_cols],
                          rotation=45, ha='right')
        ax.set_ylabel('Normalized Parameter Value')
        ax.set_title('Parallel Coordinates: Parameter Configurations')
        ax.legend(loc='upper right')

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_parameter_scatter(
        self,
        param_pairs: List[Tuple[str, str]],
        figsize: Tuple[int, int] = (14, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot scatter plots for parameter pairs colored by objective value.

        Args:
            param_pairs: List of (param1, param2) tuples to plot
            figsize: Figure size
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure
        """
        n_plots = len(param_pairs)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]

        trials_df = self.result.get_trials_dataframe()
        best_idx = trials_df['value'].idxmax()

        for ax, (p1, p2) in zip(axes, param_pairs):
            col1 = f'params_{p1}' if not p1.startswith('params_') else p1
            col2 = f'params_{p2}' if not p2.startswith('params_') else p2

            x = trials_df[col1].values
            y = trials_df[col2].values
            z = trials_df['value'].values

            sc = ax.scatter(x, y, c=z, cmap='RdYlGn', s=60, alpha=0.8,
                           edgecolor='black', linewidth=0.5)

            # Check if log scale is appropriate
            if col1 in self.result.bo_params.search_space:
                _, _, log_scale = self.result.bo_params.search_space[p1]
                if log_scale:
                    ax.set_xscale('log')
            if col2 in self.result.bo_params.search_space:
                _, _, log_scale = self.result.bo_params.search_space[p2]
                if log_scale:
                    ax.set_yscale('log')

            ax.set_xlabel(p1.replace('params_', ''))
            ax.set_ylabel(p2.replace('params_', ''))
            plt.colorbar(sc, ax=ax, label='Objective')

            # Mark best point
            ax.scatter(trials_df.loc[best_idx, col1],
                      trials_df.loc[best_idx, col2],
                      s=200, marker='*', color=self._get_color('accent'),
                      edgecolor='black', linewidth=1, zorder=5, label='Best')
            ax.legend()

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_comparison_bar(
        self,
        comparison_data: Dict[str, float],
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (8, 5),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot bar chart comparing different methods/configurations.

        Args:
            comparison_data: Dictionary of {method_name: value}
            ax: Optional matplotlib axes
            figsize: Figure size if creating new figure
            save_path: Optional path to save the figure

        Returns:
            matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        methods = list(comparison_data.keys())
        values = list(comparison_data.values())

        # Assign colors
        colors = [
            self._get_color('secondary'),
            self._get_color('warning'),
            self._get_color('success')
        ]
        while len(colors) < len(methods):
            colors.append(self._get_color('neutral'))

        bars = ax.bar(methods, values, color=colors[:len(methods)],
                     alpha=0.85, edgecolor='black', linewidth=1.2)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=11)

        ax.set_ylabel('Value')
        ax.set_title('Performance Comparison')
        ax.set_ylim(0, max(values) * 1.15)

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_all(
        self,
        output_dir: str,
        prefix: str = "bo",
        param_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> None:
        """
        Generate and save all standard plots.

        Args:
            output_dir: Directory to save plots
            prefix: Prefix for output files
            param_pairs: Optional list of parameter pairs for scatter plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Optimization history
        fig = self.plot_optimization_history()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{prefix}_optimization_history.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Parameter importances
        try:
            fig = self.plot_param_importances()
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{prefix}_param_importances.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Could not plot param importances: {e}")

        # Parallel coordinates
        fig = self.plot_parallel_coordinates()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{prefix}_parallel_coordinates.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Parameter scatter plots
        if param_pairs:
            fig = self.plot_parameter_scatter(param_pairs)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{prefix}_parameter_scatter.png',
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

        print(f"All plots saved to {output_dir}/")
