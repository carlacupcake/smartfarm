"""ga_bounds.py."""
import numpy as np
from pydantic import BaseModel, Field


class DesignSpaceBounds(BaseModel):
    """
    DesignSpaceBounds class to hold lower and upper bounds for design variables.

    TODO add doc string
    """
    lower_bounds: np.ndarray = Field(
        default_factory=np.ndarray,
        description="Lower bounds for properties of materials considered in the optimization."
    )
    upper_bounds: np.ndarray = Field(
        default_factory=np.ndarray,
        description="Upper bounds for properties of materials considered in the optimization."
    )

    class Config:
        arbitrary_types_allowed = True
        schema = {
            "example": {
                "lower_bounds": np.array([100, 0.5, 700, 100]), # irrig_freq, irrig_amt, fert_freq, fert_amt
                "upper_bounds": np.array([140, 2, 750, 130])
            }
        }
