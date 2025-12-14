# model_carrying_capacities.py
from pydantic import BaseModel, Field, PositiveFloat


class ModelCarryingCapacities(BaseModel):
    """
    Class to hold the carrying capacities used in the model.
    """

    kh: PositiveFloat = Field(
        default=3.0,
        description="Carrying capacity of height (m)."
    )
    kA: PositiveFloat = Field(
        default=0.65,
        description="Carrying capacity of leaf area (m^2)."
    )
    kN: PositiveFloat = Field(
        default=20,
        description="Carrying capacity of number of leaves (unitless)."
    )
    kc: PositiveFloat = Field(
        default=1000,
        description="Carrying capacity of number flower spikelets (unitless)."
    )
    kP: PositiveFloat = Field(
        default=0.25,
        description="Carrying capacity of fruit biomass (kg)."
    )
