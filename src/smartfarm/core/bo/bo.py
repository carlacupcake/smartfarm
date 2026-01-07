# bo.py
"""
Bayesian Optimization wrapper using Optuna's TPE sampler.
"""
import json
import numpy as np
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna
from optuna.samplers import TPESampler

from .bo_params import BOParams


@dataclass
class BOResult:
    """
    Container for Bayesian Optimization results.
    """
    study: optuna.Study
    best_value: float
    best_params: Dict[str, Any]
    best_trial: optuna.Trial
    n_trials: int
    elapsed_time: float
    bo_params: BOParams

    def get_trials_dataframe(self):
        """Get trials as a pandas DataFrame."""
        return self.study.trials_dataframe()

    def get_sorted_trials(self, ascending: bool = False):
        """Get trials sorted by value."""
        df = self.get_trials_dataframe()
        return df.sort_values('value', ascending=ascending)

    def save(self, output_dir: str, prefix: str = "bo") -> None:
        """
        Save results to disk.

        Args:
            output_dir: Directory to save results
            prefix: Prefix for output files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save study object
        with open(f'{output_dir}/{prefix}_study.pkl', 'wb') as f:
            pickle.dump(self.study, f)

        # Save best params as JSON
        result_dict = {
            'best_value': self.best_value,
            'best_params': self.best_params,
            'best_trial_attrs': dict(self.best_trial.user_attrs),
            'n_trials': self.n_trials,
            'elapsed_time': self.elapsed_time,
        }
        with open(f'{output_dir}/{prefix}_results.json', 'w') as f:
            json.dump(result_dict, f, indent=2)

        # Save trials DataFrame
        self.get_trials_dataframe().to_csv(
            f'{output_dir}/{prefix}_trials.csv', index=False
        )

    @classmethod
    def load(cls, output_dir: str, prefix: str = "bo",
             bo_params: Optional[BOParams] = None) -> "BOResult":
        """
        Load results from disk.

        Args:
            output_dir: Directory containing saved results
            prefix: Prefix used when saving
            bo_params: Optional BOParams (if not provided, uses defaults)

        Returns:
            BOResult object
        """
        # Load study
        with open(f'{output_dir}/{prefix}_study.pkl', 'rb') as f:
            study = pickle.load(f)

        # Load JSON for elapsed time
        with open(f'{output_dir}/{prefix}_results.json', 'r') as f:
            result_dict = json.load(f)

        return cls(
            study=study,
            best_value=study.best_value,
            best_params=study.best_params,
            best_trial=study.best_trial,
            n_trials=len(study.trials),
            elapsed_time=result_dict.get('elapsed_time', 0.0),
            bo_params=bo_params or BOParams()
        )


class BayesianOptimization:
    """
    Bayesian Optimization using Optuna's TPE sampler.

    This class provides a clean interface for running BO with:
    - Configurable search space via BOParams
    - Progress callbacks
    - Result saving/loading
    - Support for user-defined objective functions
    """

    def __init__(self, bo_params: BOParams):
        """
        Initialize Bayesian Optimization.

        Args:
            bo_params: BOParams object defining the search space and settings
        """
        self.bo_params = bo_params
        self.study: Optional[optuna.Study] = None
        self.result: Optional[BOResult] = None

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest parameters based on the search space definition.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested parameter values
        """
        params = {}

        # Float parameters
        for name, (low, high, log_scale) in self.bo_params.search_space.items():
            params[name] = trial.suggest_float(name, low, high, log=log_scale)

        # Integer parameters
        for name, (low, high) in self.bo_params.integer_params.items():
            params[name] = trial.suggest_int(name, low, high)

        return params

    def create_objective(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], float],
        user_attrs_fn: Optional[Callable[[Any], Dict[str, Any]]] = None
    ) -> Callable[[optuna.Trial], float]:
        """
        Create an Optuna objective function from a user-defined evaluator.

        Args:
            evaluate_fn: Function that takes params dict and returns objective value
            user_attrs_fn: Optional function to extract user attributes from result

        Returns:
            Optuna-compatible objective function
        """
        def objective(trial: optuna.Trial) -> float:
            # Suggest parameters
            params = self._suggest_params(trial)

            try:
                # Evaluate
                result = evaluate_fn(params)

                # Handle case where evaluate_fn returns (value, extra_data)
                if isinstance(result, tuple):
                    value, extra_data = result
                    if user_attrs_fn:
                        for key, val in user_attrs_fn(extra_data).items():
                            trial.set_user_attr(key, val)
                else:
                    value = result

                return value

            except Exception as e:
                trial.set_user_attr('error', str(e))
                # Return very bad value based on direction
                if self.bo_params.direction == "maximize":
                    return -1e10
                else:
                    return 1e10

        return objective

    def run(
        self,
        objective: Callable[[optuna.Trial], float],
        callback: Optional[Callable[[optuna.Study, optuna.Trial], None]] = None,
        show_progress: bool = True,
        verbose: bool = True
    ) -> BOResult:
        """
        Run Bayesian Optimization.

        Args:
            objective: Optuna objective function
            callback: Optional callback called after each trial
            show_progress: Whether to show progress bar
            verbose: Whether to print progress messages

        Returns:
            BOResult containing optimization results
        """
        # Create sampler
        sampler = TPESampler(
            n_startup_trials=self.bo_params.n_startup_trials,
            seed=self.bo_params.seed
        )

        # Create study
        self.study = optuna.create_study(
            direction=self.bo_params.direction,
            sampler=sampler,
            study_name=self.bo_params.study_name
        )

        if verbose:
            print(f"Starting Bayesian Optimization with {self.bo_params.n_trials} trials...")
            print(f"  Startup trials: {self.bo_params.n_startup_trials}")
            print(f"  Direction: {self.bo_params.direction}")
            print()

        # Default callback for progress
        callbacks = []
        if callback:
            callbacks.append(callback)

        if verbose and not callback:
            def default_callback(study, trial):
                if trial.number % 10 == 0:
                    print(f"Trial {trial.number}: value={trial.value:.2f}, "
                          f"best={study.best_value:.2f}")
            callbacks.append(default_callback)

        # Run optimization
        t_start = time.time()
        self.study.optimize(
            objective,
            n_trials=self.bo_params.n_trials,
            callbacks=callbacks if callbacks else None,
            show_progress_bar=show_progress
        )
        elapsed = time.time() - t_start

        if verbose:
            print(f"\nOptimization complete in {elapsed/60:.1f} minutes")
            print(f"Best value: {self.study.best_value:.4f}")

        # Create result
        self.result = BOResult(
            study=self.study,
            best_value=self.study.best_value,
            best_params=self.study.best_params,
            best_trial=self.study.best_trial,
            n_trials=len(self.study.trials),
            elapsed_time=elapsed,
            bo_params=self.bo_params
        )

        return self.result

    def get_param_importances(self) -> Dict[str, float]:
        """
        Get parameter importances using fANOVA.

        Returns:
            Dictionary mapping parameter names to importance values
        """
        if self.study is None:
            raise ValueError("No study available. Run optimization first.")

        from optuna.importance import get_param_importances
        return get_param_importances(self.study)
