"""
Object containing refinement results and functions that implement the workflow.
"""

from abc import abstractmethod
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, TypedDict, cast
from typing_extensions import Self

from .Data2D import Data2D, Data2DView
from .Expression import Expression
from .GeneSet import GeneSet
from .Phenotypes import Phenotype, Phenotypes
from .Utils import compute_information_coefficient, run_ssgsea_parallel, ssGSEAResult
from .version import __version___


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
	#_A: A_Matrix
	_A: Data2DView[Expression]
	_nmf: NMF
	_W: NMF_W_Matrix
	_H: NMF_H_Matrix

	def __init__(
		self,
		#A: A_Matrix,
		A: Data2DView[Expression],
		k: int,
		rng: np.random.Generator
	) -> None:
		"""
		Instantiate an instance of a single NMF solution for one inner
		iteration. 

		Parameters
		----------
		`A` : `Data2DView[Expression]`
			The `A` matrix to decompose, a subset of the original expression. 

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
			index = self.A.row_names
		)

		self._H = NMF_H_Matrix.from_array(
			self._nmf.components_,
			columns = self.A.col_names
		)

	@property
	def A(self) -> Data2DView[Expression]: return self._A

	@property
	def W(self) -> NMF_W_Matrix: return self._W

	@property
	def H(self) -> NMF_H_Matrix: return self._H

	@property
	def k(self) -> int: return int(self._nmf.n_components_)


class GeneComponentIC(Data2D):
	def __init__(
		self, 
		nmf_result: NMFResult,
		outer_num: int,
		inner_num: int
	) -> None:
		n_genes: int = len(nmf_result.W.row_names)
		n_components: int = nmf_result.k

		arr = np.empty(
			shape = (n_genes, n_components)
		)

		for i, gene_name in enumerate(nmf_result.A.row_names):
			
			for j, component_name in enumerate(nmf_result.H.row_names):

				A_gene_vec: List[float] = nmf_result.A.subset(
					row_names = [gene_name]
				).squeeze()

				H_comp_vec: List[float] = nmf_result.H.subset(
					row_names = [component_name]
				).squeeze()

				arr[i, j] = compute_information_coefficient(
					A_gene_vec,
					H_comp_vec
				)

		df = pd.DataFrame(
			arr,
			index = nmf_result.A.row_names,
			columns = [
				f"k{nmf_result.k}_o{outer_num}_i{inner_num}_c{h_indx}"
				for h_indx in nmf_result.H.row_names
			]
		)

		super().__init__(df)

	@property
	def data_name(self) -> str: return "gene-component ICs"

	@property
	def row_title(self) -> str: return "gene"

	@property
	def col_title(self) -> str: return "component"


class CombinedGeneComponentIC(Data2D):
	@classmethod
	def from_refinement(
		cls,
		k: int,
		refinement: "Refinement"
	) -> "CombinedGeneComponentIC":
		"""
		"""
		gene_comp_ics: List[pd.DataFrame] = []

		for outer_iteration in refinement.iterations[k]:
			for inner_iteration in outer_iteration:

				gene_comp_ics.append(
					inner_iteration.gene_component_IC.data
				)

		gene_comp_ics_df = pd.concat(gene_comp_ics, axis = 1)

		return CombinedGeneComponentIC(gene_comp_ics_df)

	@property
	def data_name(self) -> str: return "all gene-component ICs"

	@property
	def row_title(self) -> str: return "gene"

	@property
	def col_title(self) -> str: return "component"


class InnerIteration:
	## Inputs
	_train_expr: Data2DView[Expression] 
	_gs: GeneSet
	_i: int ## external outer index
	_j: int ## external inner index
	_k: int
	_rng: np.random.Generator
	_max_tries: int

	## Intermediates + results
	_gen_expr: Data2DView[Expression]
	_A: Data2DView[Expression]
	_nmf_result: NMFResult
	_gene_comp_ic: GeneComponentIC

	def __init__(
		self,
		training_expression: Data2DView[Expression],
		input_gene_set: GeneSet,
		i: int,
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
		`training_expression` : `Data2DView[Expression]`
			Subset of `Expression` object subet to the 2/3rds training samples (see
			documentation/paper). 

		`input_gene_set` : `GeneSet`
			Gene set to be refined. 

		`i` : `int`
			The index for the outer loop iteration. 

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
		self._i = i
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

		self._gene_comp_ic = GeneComponentIC(
			self._nmf_result,
			self._i,
			self._j
		)

	def _get_A(self) -> Data2DView[Expression]:
		"""
		Returns the `A` matrix for this interation, i.e., the generation matrix
		subset to the gene set genes. 

		Returns
		-------
		`Expression`
			The `A` matrix containing the gene set genes and the generation 
			samples. 
		"""
		shared_genes = list(set(self._gs.genes).intersection(self._gen_expr.row_names))

		return self._gen_expr.subset(row_names = shared_genes)


	@property
	def k(self) -> int: return self._k

	@property
	#def A(self) -> A_Matrix: return self._A
	def A(self) -> Data2DView[Expression]: return self._A

	@property
	def W(self) -> NMF_W_Matrix: return self._nmf_result.W

	@property
	def H(self) -> NMF_H_Matrix: return self._nmf_result.H

	@property
	def gene_component_IC(self) -> GeneComponentIC: return self._gene_comp_ic

	@property
	#def training_expression(self) -> Expression: return self._train_expr
	def training_expression(self) -> Data2DView[Expression]: 
		return self._train_expr

	@property
	#def generating_expression(self) -> Expression: return self._gen_expr
	def generating_expression(self) -> Data2DView[Expression]:
		return self._gen_expr

class ComponentCluster:
	_k: int
	_cluster_number: int
	_cutoff: float
	
	_component_names: List[str]
	_gene_set: GeneSet

	def __init__(
		self,
		refinement: "Refinement",
		comb_gene_comp_ics: CombinedGeneComponentIC,
		k: int,
		cluster_number: int
	) -> None:
		"""
		"""
		self._k = k
		self._cluster_number = cluster_number
		self._cutoff = refinement._cutoff

		## Subset gene/comp IC
		cluster_labels = pd.Series(
			refinement.kmeans[self._k].labels_,
			index = comb_gene_comp_ics.col_names
		)

		cluster_labels = cluster_labels[cluster_labels == self._cluster_number]

		cluster_gene_comp_ic = self._subset_gene_comp_ic(
			comb_gene_comp_ics,
			cluster_labels
		)

		self._component_names = cluster_gene_comp_ic.col_names

		## Generate gene set based on cutoff
		self._gene_set = self._get_component_gene_set(cluster_gene_comp_ic)

	def _subset_gene_comp_ic(
		self,
		gene_comp_ic_df: CombinedGeneComponentIC,
		cluster_labels: pd.Series
	) -> Data2DView[CombinedGeneComponentIC]:
		"""
		"""
		samples_in_cluster: List[str] = cluster_labels[
			cluster_labels == self._cluster_number
		].index.tolist()

		return gene_comp_ic_df.subset(
			column_names = samples_in_cluster
		)


	def _get_component_gene_set(
		self,
		#cluster_gene_comp_ic: CombinedGeneComponentIC
		cluster_gene_comp_ic: Data2DView[CombinedGeneComponentIC]
	) -> GeneSet:
		"""
		"""
		## Typing skipped because current version of Pandas 
		## doesn't type apply() correctly. 
		cluster_median_s = pd.Series(
			cluster_gene_comp_ic.data.apply( #type: ignore
				func = np.median,
				axis = 1
			)
		).sort_values(ascending = False)

		gene_set_genes: List[str] = cluster_median_s[
			cluster_median_s > self._cutoff
		].index.to_list()

		return GeneSet(
			f"k{self._k}_c{self._cluster_number}",
			gene_set_genes
		)

	@property
	def cluster_name(self) -> str: return f"cluster_{self._cluster_number}"

	@property
	def component_names(self) -> List[str]:
		return self._component_names

	@property
	def gene_set(self) -> GeneSet: return self._gene_set


class PhenotypeComponentIC(Data2D):
	
	class PhenCompNumericException(Exception):
		def __init__(
			self,
			e: Exception,
			ssgsea_data: Data2DView[ssGSEAResult],
			phen_vec: Data2DView[Phenotype],
			gene_set_name: str,
			phen_name: str,
			phen_feature_name: str,
			gene_set: GeneSet
		) -> None:
			"""
			"""
			ssgsea_preview = ', '.join(
				list(map(str, ssgsea_data.data.loc[gene_set_name,:].squeeze().to_list()[:10]))
			)

			phen_preview = ','.join(
				list(map(str, phen_vec.data.loc[phen_feature_name,:].squeeze().to_list()[:10]))
			)

			msg = ""
			msg += f"Unknown numerical error while computing IC between "
			msg += f"gene set '{gene_set_name}' and phenotype "
			msg += f"'{phen_feature_name}' from {phen_name}.\n"
			msg += f"First ten values of ssGSEA vector: {ssgsea_preview}.\n"
			msg += f"First ten values of phenotype vector: {phen_preview}.\n"
			msg += f"Gene set: {','.join(gene_set.genes)}\n"
			msg += f"Exception text:\n{e}"

			super().__init__(msg)

	@classmethod
	def from_refinement(
		cls,
		k: int,
		i: int, 
		phen_name: str,
		refinement: "Refinement"
	) -> "PhenotypeComponentIC":
		"""
		"""
		full_ssgsea_res_df: ssGSEAResult = refinement.ssgsea_res[k]
		test_sample_names: List[str] = refinement.test_samples[k][i]
		phen_data: Phenotype = refinement.phenotypes[phen_name]

		test_ssgsea_res = full_ssgsea_res_df.subset(
			column_names = test_sample_names
		)

		ic_array = np.empty((test_ssgsea_res.shape[0], phen_data.shape[0]))

		for x, gene_set_name in enumerate(test_ssgsea_res.row_names):
			
			for y, phen_feature_name in enumerate(phen_data.row_names):

				one_ssgsea_res = test_ssgsea_res.subset(
					row_names = [gene_set_name]
				)

				try:
					phen_vec = phen_data.subset_shared(
						one_ssgsea_res,
						shared_cols = True
					)[0].subset(row_names = [phen_feature_name])


				except Data2D.NanFilterException:
					ic_array[x, y] = np.nan
					continue

				one_ssgsea_res, phen_vec = one_ssgsea_res.subset_shared(
					phen_vec,
					shared_cols = True
				)

				try:
					ic = compute_information_coefficient(
						one_ssgsea_res.squeeze(),
						phen_vec.squeeze()
					)
				except Exception as e:
					# self._component_clusters[k].values()

					k = int(gene_set_name[1])
					comp_num = int(gene_set_name[4])
					gene_set = refinement._component_clusters[k][comp_num].gene_set
					
					raise cls.PhenCompNumericException(
						e,
						one_ssgsea_res,
						phen_vec,
						gene_set_name,
						phen_name,
						phen_feature_name,
						gene_set
					)

				ic_array[x,y] = ic

		res_df = pd.DataFrame(
			ic_array,
			index = test_ssgsea_res.row_names,
			columns = phen_data.row_names
		)

		return cls(res_df)

	@property
	def data_name(self) -> str: return "Phenotype and Component IC"

	@property
	def row_title(self) -> str: return "phenotype"

	@property
	def col_title(self) -> str: return "component"


class Refinement:
	_version: str = __version___

	_expr_path: str
	_min_counts: int
	_phen_paths: Dict[str, str]
	_gs_path: str
	_gs_name: str
	_ks: List[int]
	_n_outer: int
	_n_inner: int
	_cutoff: float
	_gsea_stat: str
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
	_component_clusters: Dict[int, Dict[int, ComponentCluster]]
	_all_ssgsea_res: Dict[int, ssGSEAResult]
	_phen_comp_ics: Dict[int, List[Dict[str, PhenotypeComponentIC]]]

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
		cutoff: float = 0.3,
		gsea_statistic: str = "auc",
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

		`cutoff` : `float`, default `0.3`
			IC cutoff for gene set inclusion.

		`gsea_statistic` : `str`
			Enrichment scoring method, must be either `"ks"` or `"auc"`

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
		self._cutoff = cutoff
		self._gsea_stat = gsea_statistic
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

		log_s += f"{' | ' * tabs} "

		log_s += f"{msg}"

		print(log_s)

	def run(self):
		"""
		"""
		self._iterations = {}
		self._test_samples = {}
		self._kmeans = {}
		self._component_clusters = {}
		self._all_ssgsea_res = {}
		self._phen_comp_ics = {}

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

				## TODO: Can rewrite test samples as Data2DView lists
				self._test_samples[k].append(
					test.col_names
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
						i,
						j,
						k,
						self._rng,
						max_downsample_tries = self._max_tries
					)

					ii.run()

					self._iterations[k][i].append(ii)

			## Concatenate components across lists

			self._log(f"Combining and clustering components (k = {k})")

			comb_gene_comp_ics = CombinedGeneComponentIC.from_refinement(
				k,
				self
			)

			## Cluster components

			self._kmeans[k] = KMeans(
				n_clusters = k,
				random_state = self._rng.integers(
					low = 1, 
					high = np.iinfo(np.int32).max
				),
				n_init = "auto"
			).fit(comb_gene_comp_ics.data.T)

			self._component_clusters[k] = {}

			for cluster_number in range(k):
				self._component_clusters[k][cluster_number] = ComponentCluster(
					self,
					comb_gene_comp_ics,
					k,
					cluster_number
				)

			self._log((
				f"Computing test set enrichment score associations with "
				f"phenotype (k = {k})"
			))

			gene_sets = (
				[self._gs] + [
					comp_clust.gene_set
					for comp_clust in self._component_clusters[k].values()
				]
			)

			#TODO: Implement parallelization and pass through n jobs
			self._log((
				f"Running ssGSEA on test scores"
			), tabs = 1)
			self._all_ssgsea_res[k] = run_ssgsea_parallel(
				self._expr,
				gene_sets,
				statistic = self._gsea_stat
			)

			self._phen_comp_ics[k] = []

			for i in range(self._n_outer):
				self._log((
					f"Comparing phenotypes and ssGSEA scores, outer iteration {i}"
				), tabs = 1)
				self._phen_comp_ics[k].append({})

				for phen_name in self._phens.phenotype_table_names:
					self._phen_comp_ics[k][i][phen_name] = (
						PhenotypeComponentIC.from_refinement(
							k, i, phen_name, self
						)
					)

	def save(
		self,
		path: str
	) -> None:
		"""
		"""
		with open(path, 'wb') as f:
			pickle.dump(self, f)

	class RefinementVersionWarning(Warning):
		def __init__(
			self,
			cls_version: str,
			file_version: str
		) -> None:
			"""
			"""
			msg = ""
			msg += f"Loading a refinement object from version {file_version} "
			msg += f"using Refinement version {cls_version}. This may result "
			msg += f"in errors or unexpected behavior."

			super().__init__(msg)

	@classmethod
	def load(
		cls,
		path: str
	) -> "Refinement":
		with open(path, 'rb') as f:
			obj = cast(Refinement, pickle.load(f))

		if obj._version != cls._version:
			raise cls.RefinementVersionWarning(cls._version, obj._version)

		return obj

	@property
	def iterations(self) -> Dict[int, List[List[InnerIteration]]]:
		return self._iterations

	@property
	def k_values(self) -> List[int]: return self._ks

	@property
	def component_clusters(self) -> Dict[int, Dict[int, ComponentCluster]]:
		return self._component_clusters

	@property
	def kmeans(self) -> Dict[int, KMeans]: return self._kmeans

	@property
	def ssgsea_res(self) -> Dict[int, ssGSEAResult]: return self._all_ssgsea_res

	@property
	def test_samples(self) -> Dict[int, List[List[str]]]: 
		return self._test_samples

	@property
	def phenotypes(self) -> Phenotypes: return self._phens

	@property
	def phenotype_component_ics(
		self
	) -> Dict[int, List[Dict[str, PhenotypeComponentIC]]]:
		return self._phen_comp_ics



