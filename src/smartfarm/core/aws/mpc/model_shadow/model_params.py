# model_params.py
import math
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, model_validator
from typing import Optional


class ModelParams(BaseModel):
    """
    Class to hold the time parameters used for in the model.
    """

    dt: PositiveFloat = Field(
        default=0.1,
        description="Size of each time step in hours."
    )
    total_time_steps: PositiveInt = Field(
        default=None,
        description="Total number of time steps in the simulation."
    )
    simulation_hours: Optional[PositiveInt] = Field(
        default=2900,
        description="Total number of hours to simulate per model run."
    )
    closed_form: bool = Field(
        default=True,
        description="Whether to use closed-form solutions " \
                    "(as opposed to Forward Euler integration) for growth calculations."
    )
    verbose: bool = Field(
        default=False,
        description="Whether to save internal variables to csv for further analysis."
    )
        
    @model_validator(mode='after')
    def _compute_total_time_steps(self):
        if self.total_time_steps is None:
            self.total_time_steps = int(math.ceil(self.simulation_hours / self.dt))
        return self
