# model_typical_disturbances.py
from pydantic import BaseModel, Field, PositiveFloat


class ModelTypicalDisturbances(BaseModel):
    """
    Class to hold the parameters used for the genetic algorithm.
    """

    typical_water: PositiveFloat = Field(
        default=28,
        description="Typical water/hour in inches/acre."
    )
    typical_fertilizer: PositiveFloat = Field(
        default=355,
        description="Typical fertilizer/hour in pounds/acre."
    )
    typical_temperature: PositiveFloat = Field(
        default=23,
        description="Typical (mean) hourly temperature in degrees Celsius."
    )
    typical_radiation: PositiveFloat = Field(
        default=1500,
        description="Typical (mean) hourly radiation in watts/m²."
    )
