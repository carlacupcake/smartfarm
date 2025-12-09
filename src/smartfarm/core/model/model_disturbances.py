# model_disturbances.py
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict


class ModelDisturbances(BaseModel):
    """
    Class to hold the arrays of environmental disturbances.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    precipitation: np.ndarray = Field(
        default=np.ones([3000]) * 0.01,
        description="Precipitation in inches."
    )
    temperature: np.ndarray = Field(
        default=np.ones([3000]) * 25,
        description="Temperature in degrees Celsius."
    )
    radiation: np.ndarray = Field(
        default=np.ones([3000]) * 580,
        description="Solar radiation in watts per square meter."
    )

    @classmethod
    def from_defaults(cls, path: str):
        """
        Create a ModelDisturbances instance from a CSV file.
        For example, use path = '../../io/inputs/hourly_prcp_rad_temp_iowa.csv'
        """
        df = pd.read_csv(path)
        return cls(
            precipitation=df["Hourly Precipitation (in)"].to_numpy(),
            radiation=df["Hourly Radiation (W/m2)"].to_numpy(),
            temperature=df["Temperature (C)"].to_numpy(),
        )
