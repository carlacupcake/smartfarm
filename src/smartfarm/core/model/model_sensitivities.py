# model_sensitivities.py
from pydantic import BaseModel, Field, PositiveFloat


class ModelSensitivities(BaseModel):
    """
    Class to hold the growth rates used in the model.
    """
    alpha: PositiveFloat = Field(
        default=3.0, # exp(-alpha * |x|) means when |x| = 1, nu = exp(-3) = 0.05
        description="Parameter for nutrient factor sensitivity to deviations of \
            cumulative divergence from expected watering/fertilization/temp/radiation."
    )
    beta: PositiveFloat = Field(
        default=0.95,
        description="Exponential moving average beta parameter for cumulative \
            divergence calculation used in nutrient factor estimation. \
            beta close to 1 => long memory; beta smaller => faster forgetting."
    )
    epsilon: PositiveFloat = Field(
        default=1e-6,
        description="Small positive number to ensure numerical stability."
    )
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
