# mpc_bounds.py
import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, Tuple

class ControlInputBounds(BaseModel):
    """
    TODO
    """
    irrigation_bounds: Tuple[Optional[float], Optional[float]] = Field(
        default_factory=lambda: (0.0, 0.7), # lower, upper in inches
        description="Lower and upper bounds for irrigation amount applied at each hour."
    )
    fertilizer_bounds: Tuple[Optional[float], Optional[float]] = Field(
        default_factory=lambda: (0.0, 12.0), # lower, upper in lbs
        description="Lower and upper bounds for fertilizer amount applied at each hour."
    )
    irrigation_amount_guess: Optional[float] = Field(
        default=0.007,
        description="Initial guess (in inches) for MPC solver for irrigation amount at each hour."
    )
    fertilizer_amount_guess: Optional[float] = Field(
        default=0.12,
        description="Initial guess (in lbs) for MPC solver for fertilizer amount at each hour."
    )
