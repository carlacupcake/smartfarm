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
        default=30,
        description="Number of days in the model predictive control horizon."
    )
    weight_irrigation: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=0.1, # scaled down to encourage actuation
        description="Economic penalty per unit of irrigation applied (in $/acre-inch); \
            used to convert water usage into a cost term in the GA objective."
    )
    weight_fertilizer: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=0.005, # scaled down to encourage actuation
        description="Economic penalty per unit of fertilizer applied (in $/lb-acre); \
            used to convert fertilizer usage into a cost term in the GA objective."
    )
    weight_height: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=100.0,
        description="Economic reward per unit of plant height (in $ per cm-acre-plant basis); \
            drives the GA to maximize early growth."
    )
    weight_leaf_area: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=100.0,
        description="Economic reward per unit of leaf area (in $ per cm²-acre-plant basis); \
            drives the GA to maximize canopy development."
    )
    weight_fruit_biomass: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=2000.0, # $4/bushel, 1 bushel is ~25.5 kg so $0.157 per kg, 28,350 plants per acre => 4450 dollar-plants per kg-acre
        description="Economic reward per unit of fruit biomass produced \
            (in $ per kg-acre-plant basis); drives the GA to maximize harvest value."
    )
    weight_water_anomaly: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=0.1, # since the other terms are in hundreds of dollars, this keeps the nutrient factor penalty on the same order
        description="Non-economic weight for nutrient factors. \
            Ensures nutrient factors stay close to 1.0 in CFTOC."
    )
    weight_fertilizer_anomaly: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=0.1, # since the other terms are in hundreds of dollars, this keeps the nutrient factor penalty on the same order
        description="Non-economic weight for nutrient factors. \
            Ensures nutrient factors stay close to 1.0 in CFTOC."
    )
    solver: str = Field(
        default="ipopt",
        description="Name of the optimization solver to use in MPC."
    )
    solver_options: Optional[Dict[str, Any]] = Field(
        default={
            "max_iter": 2000,
            "tol": 1e-6,
            "acceptable_tol": 1e-3,
            "acceptable_iter": 20,
            "bound_relax_factor": 1e-8,
            "acceptable_constr_viol_tol": 1e-4,
            "print_level": 5
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
