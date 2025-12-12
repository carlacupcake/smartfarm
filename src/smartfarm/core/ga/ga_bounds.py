# ga_bounds.py
import numpy as np
from pydantic import BaseModel, Field


class DesignSpaceBounds(BaseModel):
    """
    Container for the admissible ranges of the genetic algorithm’s design
    variables. Each member of the population must choose values within these
    lower and upper bounds during optimization.

    The bounds are stored as NumPy arrays of equal length, with each index
    corresponding to a specific decision variable (e.g., irrigation frequency,
    irrigation amount, fertilizer frequency, fertilizer amount).
    """

    lower_bounds: np.ndarray = Field(
        default_factory=np.ndarray,
        description="Lower bounds for irrigation/fertilizer frequency/amount considered in the optimization."
    )
    upper_bounds: np.ndarray = Field(
        default_factory=np.ndarray,
        description="Upper bounds for irrigation/fertilizer frequency/amount considered in the optimization."
    )

    class Config:
        arbitrary_types_allowed = True
        schema = {
            "example": {
                "lower_bounds": np.array([100, 0.5, 700, 100]), # irrig_freq, irrig_amt, fert_freq, fert_amt
                "upper_bounds": np.array([140, 2, 750, 130])
            }
        }
