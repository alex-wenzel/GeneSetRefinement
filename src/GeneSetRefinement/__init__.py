__version__ = "1.6.2"
from .Cophenetic import get_nmf_consensus_cophenetic
from .Data2D import Data2D, Data2DView
from .Expression import Expression
from .GeneSet import GeneSet, read_gmt, MSigDBQuery
from .Phenotypes import Phenotypes
from .Refinement import Refinement, kRefinement
from .InnerIteration import InnerIteration
from .GeneComponentIC import CombinedGeneComponentIC
from .Phenotypes import Phenotype
from .PhenotypeComponentIC import PhenotypeComponentIC
from .Utils import load_gct, compute_information_coefficient, ssGSEAResult