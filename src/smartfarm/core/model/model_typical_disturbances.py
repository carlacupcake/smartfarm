# model_typical_disturbances.py
from pydantic import BaseModel, Field, PositiveFloat


class ModelTypicalDisturbances(BaseModel):
    """
    Class to hold the parameters used for the genetic algorithm.
    """

    typical_water: PositiveFloat = Field(
        default=0.01, # 28 inches over the season, with 2900 hours in the season, that is ~0.01 inches/hour
        description="Typical water/hour in inches/acre."
    )
    typical_fertilizer: PositiveFloat = Field(
        default=0.12, # 355 pounds over the season, with 2900 hours in the season, that is ~0.12 pounds/hour
        description="Typical fertilizer/hour in pounds/acre."
    )
    typical_temperature: PositiveFloat = Field(
        default=22.82,
        description="Typical (mean) hourly temperature in degrees Celsius."
    )
    typical_radiation: PositiveFloat = Field(
        default=580,
        description="Typical (mean) hourly radiation in watts/m²."
    )
