# model_disturbances.py
import numpy as np
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
