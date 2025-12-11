# model_sensitivities.py
from pydantic import BaseModel, Field, PositiveFloat


class ModelSensitivities(BaseModel):
    """
    Class to hold the growth rates used in the model.
    """

    sigma_W: PositiveFloat = Field(
        default=30,
        description="Sensitivity of water uptake. Standard deviation of a Gaussian;" \
                    "governs delayed absorption of water (hrs))"
    )
    sigma_F: PositiveFloat = Field(
        default=300,
        description="Sensitivity of fertilizer uptake. Standard deviation of a Gaussian;" \
                    "governs delayed absorption of fertilizer (hrs))"
    )
    sigma_T: PositiveFloat = Field(
        default=30,
        description="Sensitivity of temperature uptake. Standard deviation of a Gaussian;" \
                    "governs delayed absorption of temperature (hrs))"
    )
    sigma_R: PositiveFloat = Field(
        default=30,
        description="Sensitivity of radiation uptake. Standard deviation of a Gaussian;" \
                    "governs delayed absorption of radiation (hrs))"
    )
