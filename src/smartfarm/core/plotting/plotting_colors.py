# plotting_colors.py
# reference: https://venngage.com/blog/color-blind-friendly-palette/
from cycler import cycler
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

class PlottingColors(BaseModel):
    """
    Class to hold color-blind friendly colors for plotting.
    """

    vivid_green: str = Field(
        default="#007D34"
    )
    vivid_red: str = Field(
        default="#C10020"
    )
    vivid_yellow: str = Field(
        default="#FFB300"
    )
    strong_purple: str = Field(
        default="#803E75"
    )
    vivid_orange: str = Field(
        default="#FF6800"
    )
    strong_purplish_pink: str = Field(
        default="#F6768E"
    )
    strong_blue: str = Field(
        default="#00538A"
    )
    strong_yellowish_pink: str = Field(
        default="#FF7A5C"
    )
    deep_purple: str = Field(
        default="#543879"
    )
    grayish_yellow: str = Field(
        default="#CEA261"
    )
    medium_gray: str = Field(
        default="#817066"
    )
    very_light_blue: str = Field(
        default="#A6BDD7"
    )

    def get_cycler(self) -> cycler:
        """
        Generates a Matplotlib cycler object from the defined colors.
        """
        # Extract all field values into a list of colors
        colors_list = list(self.model_dump().values())
        
        # Return a cycler object using the extracted colors
        return cycler(color=colors_list)

    def apply_as_default(self):
        """
        Applies this color palette's cycler as the default for all plots.
        """
        plt.rcParams['axes.prop_cycle'] = self.get_cycler()
