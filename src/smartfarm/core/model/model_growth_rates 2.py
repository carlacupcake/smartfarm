"""model_growth_rates.py."""
from pydantic import BaseModel, Field, PositiveFloat


class ModelGrowthRates(BaseModel):
    """
    Class to hold the growth rates used in the model.
    """

    ah: PositiveFloat = Field(
        default=0.01,
        description="Growth rate of height (1/hr)."
    )
    aA: PositiveFloat = Field(
        default=0.0105,
        description="Growth rate of leaf area (1/hr)."
    )
    aN: PositiveFloat = Field(
        default=0.011,
        description="Growth rate of number of leaves (1/hr)."
    )
    ac: PositiveFloat = Field(
        default=0.01,
        description="Growth rate of number flower spikelets (1/hr)."
    )
    aP: PositiveFloat = Field(
        default=0.005,
        description="Growth rate of fruit biomass (1/hr)."
    )
