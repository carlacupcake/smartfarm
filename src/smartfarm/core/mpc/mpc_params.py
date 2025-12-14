# ga_params.py
from typing import Annotated, Any, Dict, Optional
from pydantic import BaseModel, Field, PositiveInt, PositiveFloat


class MPCParams(BaseModel):
    """
    Container for all the parameters used by the model predictive control.
    This includes the control horizon and the economic weights that convert
    irrigation, fertilizer, and fruit biomass into a dollar-valued objective
    for optimization.
    """

    hourly_horizon: PositiveFloat = Field(
        default=10,
        description="Number of hours in the model predictive control horizon."
    )
    weight_irrigation: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=0.1, #2.0, # $2/acre-inch
        description="Economic penalty per unit of irrigation applied (in $/acre-inch); \
            used to convert water usage into a cost term in the GA objective."
    )
    weight_fertilizer: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=0.0001, #0.614, # from typical NPK ratio for corn: 230 lb/acre N ($0.68/lb), 60 P ($.56/lb), 65 K ($0.43/lb) => 1/(230 + 65 + 60) * (230 * 0.68 + 60 * 0.56 + 65 * 0.43) = $0.614 per lb-acre of fertilizer
        description="Economic penalty per unit of fertilizer applied (in $/lb-acre); \
            used to convert fertilizer usage into a cost term in the GA objective."
    )
    weight_fruit_biomass: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=4450, # $4/bushel, 1 bushel is ~25.5 kg so $0.157 per kg, 28,350 plants per acre => 4450 dollar-plants per kg-acre
        description="Economic reward per unit of fruit biomass produced \
            (in $ per kg-acre-plant basis); drives the GA to maximize harvest value."
    )
    weight_cumulative_average_water: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=1.0, # since the other terms are in hundreds of dollars, this keeps the nutrient factor penalty on the same order
        description="Non-economic weight for nutrient factors. \
            Ensures nutrient factors stay close to 1.0 in CFTOC."
    )
    weight_cumulative_average_fertilizer: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=1.0, # since the other terms are in hundreds of dollars, this keeps the nutrient factor penalty on the same order
        description="Non-economic weight for nutrient factors. \
            Ensures nutrient factors stay close to 1.0 in CFTOC."
    )
    weight_cumulative_average_temperature: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=1.0, # since the other terms are in hundreds of dollars, this keeps the nutrient factor penalty on the same order
        description="Non-economic weight for nutrient factors. \
            Ensures nutrient factors stay close to 1.0 in CFTOC."
    )
    weight_cumulative_average_radiation: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=1.0, # since the other terms are in hundreds of dollars, this keeps the nutrient factor penalty on the same order
        description="Non-economic weight for nutrient factors. \
            Ensures nutrient factors stay close to 1.0 in CFTOC."
    )
    solver: str = Field(
        default="ipopt",
        description="Name of the optimization solver to use in MPC."
    )
    solver_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional dictionary of solver options to pass to the \
            optimization solver used in MPC."
    )
    reoptimization_interval: PositiveInt = Field(
        default=1,
        description="Number of time steps between re-optimizations in MPC. \
        Choose 1 for true closed-loop MPC. Anything larger results in move-blocked MPC."
    )
    epsilon: PositiveFloat = Field(
        default=1e-6,
        description="Small positive number to ensure numerical stability."
    )
