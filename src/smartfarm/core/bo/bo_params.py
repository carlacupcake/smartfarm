# bo_params.py
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class BOParams(BaseModel):
    """
    Parameters for Bayesian Optimization.
    """

    n_trials: int = Field(
        default=100,
        description="Total number of optimization trials."
    )
    n_startup_trials: int = Field(
        default=20,
        description="Number of random trials before Bayesian sampling begins."
    )
    seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility."
    )
    direction: str = Field(
        default="maximize",
        description="Optimization direction: 'maximize' or 'minimize'."
    )
    study_name: str = Field(
        default="bayesian_optimization",
        description="Name for the Optuna study."
    )

    # Search space bounds (parameter_name -> (low, high, log_scale))
    search_space: Dict[str, Tuple[float, float, bool]] = Field(
        default={},
        description="Search space definition: {param_name: (low, high, log_scale)}."
    )

    # Integer parameters (parameter_name -> (low, high))
    integer_params: Dict[str, Tuple[int, int]] = Field(
        default={},
        description="Integer parameters: {param_name: (low, high)}."
    )

    class Config:
        arbitrary_types_allowed = True


class MPCWeightSearchSpace:
    """
    Default search space for MPC weight tuning.
    """

    @staticmethod
    def get_default() -> BOParams:
        """Get default BOParams for MPC weight tuning."""
        return BOParams(
            n_trials=100,
            n_startup_trials=20,
            seed=42,
            direction="maximize",
            study_name="mpc_weight_tuning",
            search_space={
                # Cost weights (log scale)
                "weight_irrigation": (0.001, 10.0, True),
                "weight_fertilizer": (0.0001, 1.0, True),
                # Reward weights (linear scale)
                "weight_height": (10.0, 1000.0, False),
                "weight_leaf_area": (10.0, 1000.0, False),
                "weight_fruit_biomass": (100.0, 10000.0, False),
                # Anomaly penalties (log scale)
                "weight_water_anomaly": (0.001, 10.0, True),
                "weight_fertilizer_anomaly": (0.001, 10.0, True),
            },
            integer_params={
                "daily_horizon": (3, 14),
            }
        )

    @staticmethod
    def get_robust() -> BOParams:
        """Get BOParams for robust multi-scenario MPC weight tuning."""
        params = MPCWeightSearchSpace.get_default()
        params.n_trials = 50  # Fewer trials since each is more expensive
        params.n_startup_trials = 15
        params.seed = 123
        params.study_name = "mpc_robust_weight_tuning"
        return params
