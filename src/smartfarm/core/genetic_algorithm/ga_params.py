"""ga_params.py."""
from typing import Annotated
from pydantic import BaseModel, Field, PositiveInt


class GeneticAlgorithmParams(BaseModel):
    """
    Class to hold the parameters used for the genetic algorithm.
    """

    num_parents: PositiveInt = Field(
        default=10,
        description="Number of parent members to retain in each generation."
    )
    num_kids: PositiveInt = Field(
        default=10,
        description="Number of children to produce from the parent members."
    )
    num_generations: PositiveInt = Field(
        default=100,
        description="Total number of generations to simulate in the genetic algorithm."
    )
    num_members: PositiveInt = Field(
        default=200,
        description="Total number of members in each generation of the population."
    )
    weight_height: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=1.0,
        description="TODO add doc string"
    )
    weight_fruit_biomass: Annotated[float, Field(strict=True, ge=0)] = Field(
        default=4450, # $4/bushel, 1 bushel is ~25.5 kg so $0.157 per kg, 28,350 plants per acre => 4450 dollar-plants per kg-acre
        description="TODO add doc string"
    )
