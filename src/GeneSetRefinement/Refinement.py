"""
Object containing refinement results and functions that implement the workflow.
"""

from abc import abstractmethod
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from typing import Dict, List, Optional
from typing_extensions import Self

from .Data2D import Data2D
from .Expression import Expression
from .GeneSet import GeneSet
from .Phenotypes import Phenotypes
from .Utils import compute_information_coefficient, run_ssgsea_parallel, ssGSEAResult


class A_Matrix(Expression):
	@classmethod
	def _from_data2d(
		cls,
		data2d: "Expression | A_Matrix"
	) -> "A_Matrix":
		"""
		"""
		return cls(
			data2d.data,
			data2d.min_counts
		)

	@classmethod
	def from_expression(cls, expr: Expression) -> "A_Matrix":
		return cls._from_data2d(expr)

	@classmethod
	def from_A_matrix(cls, a_matrix: "A_Matrix") -> "A_Matrix":
		return cls._from_data2d(a_matrix)

	@property
	def data_name(self) -> str: return "A matrix"


class NMFResultMatrix(Data2D):
	@classmethod
	def _np2df(
		cls,
		arr: np.ndarray,
		index: Optional[List[str]] = None,
		columns: Optional[List[str]] = None
	) -> pd.DataFrame:
		df_index = index
		df_columns = columns

		if df_index is None:
			df_index = [
				str(i) for i in range(arr.shape[0])
			]

		if df_columns is None:
			df_columns = [
				str(i) for i in range(arr.shape[1])
			]

		df = pd.DataFrame(arr, index = df_index, columns = df_columns)

		return df

	@classmethod
	@abstractmethod
	def from_array(
		cls,
		arr: np.ndarray,
		index: Optional[List[str]] = None,
		columns: Optional[List[str]] = None
	) -> Self:
		pass


class NMF_W_Matrix(NMFResultMatrix):
	@classmethod
	def from_array(
		cls,
		arr: np.ndarray,
		index: Optional[List[str]] = None,
		columns: Optional[List[str]] = None
	) -> "NMF_W_Matrix":
		"""
		"""
		df = cls._np2df(
			arr, 
			index = index, 
			columns = columns
		)

		return NMF_W_Matrix(df)

	@property
	def data_name(self) -> str: return "W matrix"

	@property
	def row_title(self) -> str: return "gene"

	@property
	def col_title(self) -> str: return "component"


class NMF_H_Matrix(NMFResultMatrix):
	@classmethod
	def from_array(
		cls,
		arr: np.ndarray,
		index: Optional[List[str]] = None,
		columns: Optional[List[str]] = None
	) -> "NMF_H_Matrix":
		"""
		"""
		df = cls._np2df(
			arr,
			index = index,
			columns = columns
		)

		return NMF_H_Matrix(df)

	@property
	def data_name(self) -> str: return "H matrix"

	@property
	def row_title(self) -> str: return "component"

	@property
	def col_title(self) -> str: return "sample"


class NMFResult:
	_A: A_Matrix
	_nmf: NMF
	_W: NMF_W_Matrix
	_H: NMF_H_Matrix

	def __init__(
		self,
		A: A_Matrix,
		k: int,
		rng: np.random.Generator
	) -> None:
		"""
		Instantiate an instance of a single NMF solution for one inner
		iteration. 

		Parameters
		----------
		`A` : `Expression`
			The `A` matrix to decompose. 

		`k` : `int`
			The value of `k` (number of components) to use. 

		`rng` : `np.random.Generator`
			The random state as passed to the inner iteration. 
		"""
		self._A = A

		self._nmf = NMF(
			n_components = k,
			init = "random",
			solver = "mu",
			beta_loss = "kullback-leibler",
			max_iter = 2000,
			random_state = rng.integers(
				low = 1,
				high = np.iinfo(np.int32).max
			)
		)

	def run(self) -> None:
		"""
		Computes and stores the W and H matrices. 
		"""
		self._W = NMF_W_Matrix.from_array(
			self._nmf.fit_transform(self._A.array),
			index = self.A.gene_names
		)

		self._H = NMF_H_Matrix.from_array(
			self._nmf.components_,
			columns = self._A.sample_names
		)

	@property
	def A(self) -> A_Matrix: return self._A

	@property
	def W(self) -> NMF_W_Matrix: return self._W

	@property
	def H(self) -> NMF_H_Matrix: return self._H

	@property
	def k(self) -> int: return int(self._nmf.n_components_)


class GeneComponentIC(Data2D):
	def __init__(
		self, 
		nmf_result: NMFResult
	) -> None:
		n_genes: int = len(nmf_result.W.row_names)
		n_components: int = nmf_result.k

		arr = np.empty(
			shape = (n_genes, n_components)
		)

		for i, gene_name in enumerate(nmf_result.A.gene_names):
			
			for j, component_name in enumerate(nmf_result.H.row_names):
				
				A_gene_vec: List[float] = Data2D.subset(
					nmf_result.A,
					row_names = [gene_name]
				).squeeze()

				H_comp_vec: List[float] = Data2D.subset(
					nmf_result.H,
					row_names = [component_name]
				).squeeze()

				arr[i, j] = compute_information_coefficient(
					A_gene_vec,
					H_comp_vec
				)

		df = pd.DataFrame(
			arr,
			index = nmf_result.A.gene_names,
			columns = nmf_result.H.row_names
		)

		super().__init__(df)

	@property
	def data_name(self) -> str: return "gene-component ICs"

	@property
	def row_title(self) -> str: return "gene"

	@property
	def col_title(self) -> str: return "component"

class InnerIteration:
	## Inputs
	_train_expr: Expression 
	_gs: GeneSet
	_j: int ## external index
	_k: int
	_rng: np.random.Generator
	_max_tries: int

	## Intermediates + results
	_gen_expr: Expression
	_A: A_Matrix
	_nmf_result: NMFResult
	_gene_comp_ic: GeneComponentIC

	def __init__(
		self,
		training_expression: Expression,
		input_gene_set: GeneSet,
		j: int,
		k: int,
		rng: np.random.Generator,
		max_downsample_tries: int = 100
	) -> None:
		"""
		Instantiate a single inner iteration to prepare for NMF and 
		gene-component IC computations. This function also generates the
		"generate" subset. 

		Parameters
		----------
		`training_expression` : `Expression`
			`Expression` object subet to the 2/3rds training samples (see
			documentation/paper). 

		`input_gene_set` : `GeneSet`
			Gene set to be refined. 

		`j` : `int`
			The index for this inner loop iteration accessed from the outer 
			loop. 

		`k` : `int`
			The value of `k` to use for NMF. 

		`rng` : `np.random.Generator`
			The RNG object originally instantiated by the `Refinement` instance
			that is creating this `InnerIteration`. 

		`max_downsample_tries` : `int`, default `100`
			The number of times to keep trying to downsample a matrix without
			generating rows of all zeros before throwing an error. 
		"""
		self._train_expr = training_expression
		self._gs = input_gene_set
		self._j = j
		self._k = k
		self._rng = rng
		self._max_tries = max_downsample_tries

		self._gen_expr = self._train_expr.subset_random_samples(
			0.5,
			self._rng,
			return_both = False,
			max_tries = self._max_tries
		)

	def run(
		self
	) -> None:
		"""
		Driver function. 
		"""
		self._A = self._get_A()

		self._nmf_result = NMFResult(
			self._A,
			self._k,
			self._rng
		)
		self._nmf_result.run()

		self._gene_comp_ic = GeneComponentIC(self._nmf_result)

	def _get_A(self) -> A_Matrix:
		"""
		Returns the `A` matrix for this interation, i.e., the generation matrix
		subset to the gene set genes. 

		Returns
		-------
		`Expression`
			The `A` matrix containing the gene set genes and the generation 
			samples. 
		"""
		return Data2D.subset(
			A_Matrix.from_expression(
				self._gen_expr
			),
			row_names = self._gs.genes,
			column_names = []
		)

	@property
	def k(self) -> int: return self._k

	@property
	def A(self) -> A_Matrix: return self._A

	@property
	def W(self) -> NMF_W_Matrix: return self._nmf_result.W

	@property
	def H(self) -> NMF_H_Matrix: return self._nmf_result.H

	@property
	def gene_component_IC(self) -> GeneComponentIC: return self._gene_comp_ic


class Refinement:
	_expr_path: str
	_min_counts: int
	_phen_paths: Dict[str, str]
	_gs_path: str
	_gs_name: str
	_ks: List[int]
	_n_outer: int
	_n_inner: int
	_seed: int
	_normalize: bool
	_max_tries: int

	_verbose: bool

	_expr: Expression
	_phens: Phenotypes
	_gs: GeneSet
	_rng: np.random.Generator

	## Dict keys are values of k
	_iterations: Dict[int, List[List[InnerIteration]]]
	_test_samples: Dict[int, List[List[str]]]
	_kmeans: Dict[int, KMeans]

	def __init__(
		self,
		expression_gct_path: str,
		phenotype_paths: Dict[str, str],
		input_gmt_path: str,
		input_gene_set_name: str,
		k_values: List[int],
		n_outer_iterations: int = 10,
		n_inner_iterations: int = 50,
		min_total_gene_counts: int = 5,
		random_seed: int = 49,
		normalize_expression: bool = True,
		max_downsample_tries: int = 100,
		verbose = False
	) -> None:
		"""
		Provide all inputs and parameters prior to running refinement. 

		Parameters
		----------
		`expression_gct_path` : `str`
			Path to a GCT file containing expression data for the multi-omics
			compendium being used for refinement. 

		`phenotype_paths` : `dict` of `str` to `str`
			Dictionary where keys are table names and values are paths to GCT
			files for phenotype data in the multi-omics compendium. 

		`input_gmt_path` : `str`
			Path to a GMT file containing the gene set to be refined. May 
			contain more than one gene set, but only one will be used, see
			`input_gene_set_name`. 

		`input_gene_set_name` : `str`
			Name of the gene set to refine, must be one of the row names in 
			`input_gmt_path`. 

		`k_values` : `list` of `int`
			The values of `k` in which to check for an optimal solution. 

		`n_outer_iterations` : `int`, default `10`
			Number of outer loop iterations to perform for refinement, see
			paper/documentation. 

		`n_inner_iterations` : `int`, default `50`
			Number of inner loop iterations to perform for refinement, see
			paper/documentation. 

		`min_total_gene_counts` : `int`, default `5`
			The minimum number of total counts for a gene across all samples
			in the GCT file at `expression_gct_path`. Genes with fewer 
			counts will be removed prior to normalization and refinement. 

		`random_seed` : `int`, default `49`
			The random seed to use for instantiating Numpy's default RNG. 

		`normalize_expression` : `bool`, default `True`
			If `True`, perform dense rank normalization and scale each sample
			to 10,000. For each sample `x` in the GCT file at 
			`expression_gct_path`,

				`10000 * (x - min(x)) / (max(x) - min(x))`

		`max_downsample_tries` : `int`, default `100`
			The number of times to keep trying to downsample a matrix without
			generating rows of all zeros before throwing an error. 

		`verbose` : `bool`, default `False`
		"""
		## Save parameters and paths
		self._expr_path = expression_gct_path
		self._min_counts = min_total_gene_counts
		self._phen_paths = phenotype_paths
		self._gs_path = input_gmt_path
		self._gs_name = input_gene_set_name
		self._ks = k_values
		self._n_outer = n_outer_iterations
		self._n_inner = n_inner_iterations
		self._seed = random_seed
		self._normalize = normalize_expression
		self._max_tries = max_downsample_tries

		self._verbose = verbose

		## Process expression
		self._log(f"Loading gene expression file {self._expr_path}...")
		self._expr = Expression.from_gct(
			self._expr_path, 
			min_counts = self._min_counts
		)
		self._log("done", tabs = 1)

		if self._normalize:
			self._log(f"Normalizing expression...")
			self._expr.normalize()
			self._log("done", tabs = 1)

		else:
			self._log((
				f"WARNING: Skipping normalization because "
				f"`normalize_expression` is `False`."
			))

		## Process phenotypes
		self._log("Loading phenotype tables...")
		self._phens = Phenotypes.from_gcts(self._phen_paths)
		self._log("done", tabs = 1)

		## Process input gene set
		self._log(f"Loading gene sets from {self._gs_path}...")
		gmt = GeneSet.from_gmt(self._gs_path)
		self._log("done", tabs = 1)

		self._log(f"Selecting {self._gs_name} to refine.")
		self._gs = gmt[self._gs_name]

		## Process RNG
		self._rng = np.random.default_rng()

	def _log(
		self,
		msg: str,
		tabs: int = 0,
		timestamp_format: str = "[%b %d, %Y %H:%M:%S]"
	) -> None:
		"""
		Internal logging function, checks `self._verbose` on each print
		to prevent writing tons of `if`s throughout. Also allows for variable
		indenting via the `tabs` option. 

		Parameters
		----------
		`msg` : `str`
			The message to print. 

		`tabs` : `int`, default 0
			Number of tabs to use as a prefix for `msg` following the timestamp. 

		`timestamp_format` : `str`
			Format string for `datetime.strftime()`
		"""
		if not self._verbose:
			return

		log_s = ""
		now = datetime.now()

		log_s += f"{now.strftime(timestamp_format)}"

		log_s += f"{'| ' * tabs} "

		log_s += f"{msg}"

	@property
	def k_values(self) -> List[int]: return self._ks

	def run(self):
		"""
		"""
		self._iterations = {}
		self._test_samples = {}
		self._kmeans = {}

		for k in self._ks:
			## Generate components and gene/component ICs

			self._iterations[k] = []
			self._test_samples[k] = []

			for i in range(self._n_outer):
				self._iterations[k].append([])

				train, test = self._expr.subset_random_samples(
					0.67,
					rng = self._rng,
					return_both = True,
					no_zero_rows = True,
					max_tries = self._max_tries
				)

				self._test_samples[k].append(
					test.sample_names
				)

				for j in range(self._n_inner):
					self._log(
						f"Running refinement component generation ("
						f"k = {k} | "
						f"outer = {i+1}/{self._n_outer} | "
						f"inner = {j+1}/{self._n_inner})"
					)


					ii = InnerIteration(
						train,
						self._gs,
						j,
						k,
						self._rng,
						max_downsample_tries = self._max_tries
					)

					ii.run()

					self._iterations[k][i].append(ii)

			## Concatenate components across lists

			self._log(f"Combining and clustering components (k = {k})")

			gene_comp_ics: List[pd.DataFrame] = []

			for outer_iteration in self._iterations[k]:
				
				for inner_iteration in outer_iteration:

					gene_comp_ics.append(
						inner_iteration.gene_component_IC.data
					)

			gene_comp_ics_df = pd.concat(gene_comp_ics, axis = 1)

			## Cluster components

			self._kmeans[k] = KMeans(
				n_clusters = k,
				random_state = self._rng.integers(
					low = 1, 
					high = np.iinfo(np.int32).max
				)
			).fit(gene_comp_ics_df.T)






