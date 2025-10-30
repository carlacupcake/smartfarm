"""model_typical_disturbances.py."""
from pydantic import BaseModel, Field, PositiveFloat


class ModelTypicalDisturbances(BaseModel):
    """
    Class to hold the parameters used for the genetic algorithm.
    """

    optimal_cumulative_water: PositiveFloat = Field(
        default=28,
        description="Optimal cumulative water in inches/acre."
    )
    optimal_cumulative_fertilizer: PositiveFloat = Field(
        default=355,
        description="Optimal cumulative fertilizer in pounds/acre."
    )
    typical_temperature: PositiveFloat = Field(
        default=23,
        description="Typical (mean) daily temperature in degrees Celsius."
    )
    typical_radiation: PositiveFloat = Field(
        default=1500,
        description="Typical (3/4 max) daily radiation in watts/m²."
    )
