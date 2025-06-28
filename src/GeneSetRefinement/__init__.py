__version__ = "1.1.0"
from .Data2D import Data2D, Data2DView
from .Expression import Expression
from .GeneSet import GeneSet, read_gmt
from .Phenotypes import Phenotypes
from .Refinement import (
	Refinement, InnerIteration, CombinedGeneComponentIC, 
	PhenotypeComponentIC, ssGSEAResult, Phenotype
)
from .Utils import load_gct, compute_information_coefficient