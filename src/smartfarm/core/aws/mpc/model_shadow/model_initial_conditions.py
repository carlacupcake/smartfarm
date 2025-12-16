# model_initial_conditions.py
from pydantic import BaseModel, Field, PositiveFloat


class ModelInitialConditions(BaseModel):
    """
    Class to hold the initial conditions for the model.
    """

    h0: PositiveFloat = Field(
        default=0.001,
        description="Initial plant height."
    )
    A0: PositiveFloat = Field(
        default=0.001,
        description="Initial leaf area per leaf in square meters."
    )
    N0: PositiveFloat = Field(
        default=0.001,
        description="Initial number of leaves."
    )
    c0: PositiveFloat = Field(
        default=0.001,
        description="Initial canopy biomass in kilograms."
    )
    P0: PositiveFloat = Field(
        default=0.001,
        description="Initial fruit biomass in kilograms."
    )
