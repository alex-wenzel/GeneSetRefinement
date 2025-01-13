from .Data2D import Data2D
from .Expression import Expression
from .GeneSet import GeneSet, read_gmt
from .Phenotypes import Phenotypes
from .Refinement import (
	Refinement, InnerIteration, A_Matrix, CombinedGeneComponentIC
)
from .Utils import load_gct, compute_information_coefficient
from .version import VERSION