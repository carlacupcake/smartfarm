# plotting_params.py
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

class PlottingParams(BaseModel):
    """
    Class to hold color-blind friendly colors for plotting.
    """

    use_latex: bool = Field(
        default=True,
        description="Enable LaTeX rendering for all text."
    )
    font_family: str = Field(
        default='serif',
        description="Main font family."
    )
    font_serif: list[str] = Field(
        default=['Computer Modern Roman'],
        description="LaTeX-like serif font."
    )
    axes_labelsize: int = Field(
        default=14
    )
    axes_titlesize: int = Field(
        default=16
    )
    figure_titlesize: int = Field(
        default=18
    )
    xtick_labelsize: int = Field(
        default=12
    )
    ytick_labelsize: int = Field(
        default=12
    )
    lines_linewidth: float = Field(
        default=1.5
    )
    axes_grid: bool = Field(
        default=True
    )
    suptitle_y_position: float = Field(
        default=0.9,
        description="The normalized Y position for fig.suptitle and tight_layout top boundary."
    )

    def __init__(self, **data):
        """
        Initialize the class and immediately apply all rcParams settings upon creation.
        """
        super().__init__(**data)
        self.apply_rcParams()


    def apply_rcParams(self):
        """
        Applies all style parameters as the default for all plots.
        """

        # Apply general style settings
        plt.rcParams['text.usetex']     = self.use_latex
        plt.rcParams['font.family']     = self.font_family
        plt.rcParams['font.serif']      = self.font_serif
        plt.rcParams['lines.linewidth'] = self.lines_linewidth
        plt.rcParams['axes.grid']       = self.axes_grid
        
        # Apply font size preferences
        plt.rcParams['axes.labelsize']   = self.axes_labelsize
        plt.rcParams['axes.titlesize']   = self.axes_titlesize
        plt.rcParams['figure.titlesize'] = self.figure_titlesize
        plt.rcParams['xtick.labelsize']  = self.xtick_labelsize
        plt.rcParams['ytick.labelsize']  = self.ytick_labelsize

        # Apply specific figure layout default (linked to the Y position constant)
        plt.rcParams['figure.subplot.top'] = self.suptitle_y_position

    
    @property
    def tight_layout_rect(self) -> list[float]:
        """
        A calculated property for the tight_layout 'rect' parameter based on the suptitle Y position.
        The format is [left, bottom, right, top].
        """
        return [0.0, 0.0, 1.0, self.suptitle_y_position]
