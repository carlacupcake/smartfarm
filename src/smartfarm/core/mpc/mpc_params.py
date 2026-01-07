# mpc_params.py
from typing import Annotated, Any, Dict, Optional
from pydantic import BaseModel, Field, PositiveInt, PositiveFloat


class MPCParams(BaseModel):
    """
    Container for all the parameters used by the model predictive control.
    This includes the control horizon and the economic weights that convert
    irrigation, fertilizer, and fruit biomass into a dollar-valued objective
    for optimization.
    """

    daily_horizon: PositiveInt = Field(
        default=9,
        description="Number of days in the model predictive control horizon."
    )
    weight_irrigation: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=0.176526, # scaled down to encourage actuation
        description="Economic penalty per unit of irrigation applied (in $/acre-inch); \
            used to convert water usage into a cost term in the objective."
    )
    weight_fertilizer: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=0.002123,
        description="Economic penalty per unit of fertilizer applied (in $/lb-acre); \
            used to convert fertilizer usage into a cost term in the objective."
    )
    weight_height: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=590.82,
        description="Economic reward per unit of plant height (in $ per cm-acre-plant basis); \
            drives the GA to maximize early growth."
    )
    weight_leaf_area: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=490.98,
        description="Economic reward per unit of leaf area (in $ per cm²-acre-plant basis); \
            drives the GA to maximize canopy development."
    )
    weight_fruit_biomass: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=1203.31,
        description="Economic reward per unit of fruit biomass produced \
            (in $ per kg-acre-plant basis); drives the GA to maximize harvest value."
    )
    weight_water_anomaly: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=2.938806,
        description="Non-economic weight for nutrient factors. \
            Ensures nutrient factors stay close to 1.0 in CFTOC."
    )
    weight_fertilizer_anomaly: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=1.207217,
        description="Non-economic weight for nutrient factors. \
            Ensures nutrient factors stay close to 1.0 in CFTOC."
    )
    solver: str = Field(
        default="ipopt",
        description="Name of the optimization solver to use in MPC."
    )
    solver_options: Optional[Dict[str, Any]] = Field(
        default={
            "tol": 1e-4,
            "acceptable_iter": 100,
            "max_iter": 500,
            "print_level": 0,
            "mu_strategy": "adaptive",
            "linear_solver": "mumps",
        },
        description="Optional dictionary of solver options to pass to the \
            optimization solver used in MPC."
    )
    reoptimization_interval: PositiveInt = Field(
        default=1, # 5
        description="Number of days between re-optimizations in MPC. \
        Choose 1 for true closed-loop MPC. Anything larger results in move-blocked MPC."
    )
    decision_interval: PositiveInt = Field(
        default=1,
        description="Time between MPC control updates (in days)."
    )
