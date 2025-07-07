"""
Object containing refinement results and functions that implement the workflow.
"""

from multiprocessing import Pool
import numpy as np
import os
import pickle
import psutil
import subprocess
from sklearn.cluster import KMeans
import sys
from typing import Dict, List, Optional, cast

from .ComponentCluster import ComponentCluster
from .Data2D import Data2DView
from .Expression import Expression
from .GeneComponentIC import CombinedGeneComponentIC
from .GeneSet import GeneSet
from .InnerIteration import InnerIteration
from .Phenotypes import Phenotypes
from .PhenotypeComponentIC import PhenotypeComponentIC
from .RefinementObject import RefinementObject
from .Utils import Log, run_ssgsea_parallel, ssGSEAResult
from pathlib import Path
__version__ = Path(__file__).with_name("__init__.py").read_text().split('\n')[0].split('=')[-1].strip('\n').strip()[1:-1]


class Refinement(RefinementObject):
	_version: str = __version__

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
	_n_proc: int

	_verbose: bool
	_log: Log

	_expr: Optional[Expression]
	_phens: Optional[Phenotypes]
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
		n_proc: int = 1,
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
		n_child_procs = len(psutil.Process(os.getpid()).children())

		n_child_threads = len(
			subprocess.check_output(
				f"ps -T {os.getpid()}",
				shell = True
			).decode("utf-8").strip('\n').split('\n')
		) - 2

		if n_child_procs > 0 or n_child_threads > 0:
			choice = input((
				f"\nWARNING: Refinement may be unstable and consume\n"
				f"excessive memory if its dependencies are allowed to\n"
				f"start threads. This Python process ({os.getpid()}) currently has\n"
				f"{n_child_procs} child processes and {n_child_threads} "
				f"child threads which\nwere likely started "
				f"by importing libraries like numpy.\nIt is strongly "
				f"recommended that you add the following lines to\n"
				f"your driver script BEFORE ANY OTHER IMPORT.\n"
				f"\n"
				f"\timport os\n"
				f"\tos.environ['OMP_NUM_THREADS'] = '1'\n"
				f"\tos.environ['MKL_NUM_THREADS'] = '1'\n"
				f"\n"
				f"If you are sure you want to continue, type YES "
				f"(otherwise type anything else or CTRL+C to exit): "
			))

			if choice != "YES":
				print("Exiting")
				sys.exit(0)


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
		self._n_proc = n_proc

		self._verbose = verbose
		self._log = Log(self._verbose)

		self._iterations = {}
		self._test_samples = {}
		self._kmeans = {}
		self._component_clusters = {}
		self._all_ssgsea_res = {}
		self._phen_comp_ics = {}

	def preprocess(self) -> None:
		"""
		All file loading, normalization, and additional field population
		"""
		## Process expresson
		self._log(f"Loading gene expression file {self._expr_path}")
		self._expr = Expression.from_gct(
			self._expr_path,
			min_counts = self._min_counts
		)
		self._log("done")
		
		if self._normalize:
			self._log(f"Normalizing expression...")
			self._expr.normalize()
			self._log("done")
		else:
			self._log((
				f"WARNING: Skipping normalization because "
				f"`normalize_expression"
			))
		
		## Process phenotypes
		self._log("Loading phenotype tables...")
		self._phens = Phenotypes.from_gcts(self._phen_paths)
		self._log("done")

		## Process input gene set
		self._log(f"Loading gene sets from {self._gs_path}...")
		gmt = GeneSet.from_gmt(self._gs_path)
		self._log("done")

		self._log(f"Selecting {self._gs_name} to refine.")
		self._gs = gmt[self._gs_name]

		## Process RNG
		self._rng = np.random.default_rng(self._seed)

	
	"""
	(InnerIteration(
		train,
		self._gs,
		i,
		j,
		k,
		seeds[j],
		Log.new_indented_log(self._log, 3),
		max_downsample_tries = self._max_tries
	),)
	"""
	
	"""
	@staticmethod
	def _ii_worker(
		ii: InnerIteration,
	) -> InnerIteration:
		###
		###
		ii.run()
		return ii
	"""

	@staticmethod
	def _ii_worker(
		train: Data2DView[Expression],
		gs: GeneSet,
		i: int,
		j: int,
		k: int,
		seed: int,
		log: Log,
		max_tries: int
	) -> InnerIteration:
		"""
		"""
		ii = InnerIteration(
			train,
			gs,
			i,
			j,
			k,
			seed,
			log,
			max_downsample_tries=max_tries
		)

		ii.run()

		return ii
		
	
	@staticmethod
	def _phen_comp_ic_worker(
		k: int,
		i: int,
		ref: "Refinement",
		log: Log
	) -> Dict[str, PhenotypeComponentIC]:
		"""
		"""
		##TODO: Need to figure out how to use log

		res_d = {}

		ref._phens = ref.assert_not_None(ref._phens)

		pref = f"(k={k}|i={i}) -"

		for phen_name in ref._phens.phenotype_table_names:
			log(f"{pref} Starting component/IC computation for '{phen_name}'...")

			res_d[phen_name] = (
				PhenotypeComponentIC.from_refinement(
					k, i, phen_name, ref
				)
			)

			log(f"{pref} done")

		return res_d

	def run(self):
		"""
		"""
		self._expr = self.assert_not_None(self._expr)

		for k in self._ks:
			## Generate components and gene/component ICs
			self._log(f"Starting refinement with k = {k}")

			self._iterations[k] = []
			self._test_samples[k] = []

			for i in range(self._n_outer):
				self._log(f"Starting outer iteration {i}...", tabs = 1)
				self._log(f"Generating train/test sets", tabs = 2)
				train, test = self._expr.subset_random_samples(
					0.67,
					rng = self._rng,
					return_both = True,
					no_zero_rows = True,
					max_tries = self._max_tries
				)
				self._log("done", tabs = 2)

				## TODO: Can rewrite test samples as Data2DView lists
				self._test_samples[k].append(
					test.col_names
				)

				seeds = self._rng.integers(
					low = 1,
					high = np.iinfo(np.int32).max,
					size = self._n_inner
				)

				self._log("Preparing inner iterations...", tabs = 2)

				#in_ii_l = [
				#	(InnerIteration(
				#		train,
				#		self._gs,
				#		i,
				#		j,
				#		k,
				#		seeds[j],
				#		Log.new_indented_log(self._log, 3),
				#		max_downsample_tries = self._max_tries
				#	),)
				in_ii_l = [
					(
						train, self._gs, i, j, k, seeds[j], 
						Log.new_indented_log(self._log, 3), self._max_tries
					)
					for j in range(self._n_inner)
				]

				self._log("done", tabs = 2)

				#with multiprocessing.get_context("spawn").Pool(self._n_proc) as pool:
				with Pool(self._n_proc) as pool:
					self._iterations[k].append(
						pool.starmap(
							self._ii_worker, 
							in_ii_l
						)
					)

				for ii in self._iterations[k][i]:
					## TEMP memory hack, make more elegant
					assert ii._train_expr is not None
					assert ii._gen_expr is not None
					assert ii._A is not None
					assert ii._nmf_result._A is not None

					ii._train_expr._data2d = self._expr
					ii._gen_expr._data2d = self._expr
					ii._A._data2d = self._expr
					ii._nmf_result._A._data2d = self._expr

			## Concatenate components across lists

			self._log(f"Combining and clustering components (k = {k})", tabs = 1)

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

			self._log("done", tabs = 1)

			self._log((
				f"Computing test set enrichment score associations with "
				f"phenotype (k = {k})"
			), tabs = 1)

			gene_sets = (
				[self._gs] + [
					comp_clust.gene_set
					for comp_clust in self._component_clusters[k].values()
				]
			)
			self._log("done", tabs = 1)

			self._log((
				f"Running ssGSEA on test scores"
			), tabs = 1)

			self._all_ssgsea_res[k] = run_ssgsea_parallel(
				self._expr,
				gene_sets,
				statistic = self._gsea_stat,
				n_job = self._n_proc
			)

			self._log("done", tabs = 1)
			
			phencomp_args = [
				(k,i,self, Log(self._verbose, base_tabs = 2)) 
				for i in range(self._n_outer)
			]

			self._log("Computing component-phenotype associations...", tabs = 1)

			with Pool(self._n_proc) as pool:
				self._phen_comp_ics[k] = (
					pool.starmap(self._phen_comp_ic_worker, phencomp_args)
				)

			self._log("done", tabs = 1)
	
	def save(
		self,
		path: str,
		remove_inputs: bool = True
	) -> None:
		"""
		Saves a standard dictionary containing class field data in a 
		Python pickle file. By default, input gene expression and phenotype
		data are removed. 
		"""
		if remove_inputs:
			save_iters = self._iterations
			save_expr = self._expr
			save_phens = self._phens

			self._expr = None 
			self._phens = None

			for k in self._ks:
				for i in range(self._n_outer):
					for j in range(self._n_inner):
						ii = self._iterations[k][i][j]

						ii._A = None
						ii._nmf_result._A = None
						ii._train_expr = None
						ii._gen_expr = None

			with open(path, 'wb') as f:
				pickle.dump(self, f)

			self._iterations = save_iters
			self._expr = save_expr
			self._phens = save_phens

		else:
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
	
	def to_gmt(
		self,
		out_path: str
	) -> None:
		"""
		Creates a GMT with the original gene set followed by each of the
		component gene sets. Will not run if refinement hasn't been run. 

		Parameters
		----------
		`out_path` : `str`
			The path to use for writing the new GMT
		"""
		gmt_rows_l: List[str] = []

		for k in self._ks:
			for comp_num in range(k):
				try:
					k_gs = self._component_clusters[k][comp_num].gene_set
				except AttributeError:
					raise RuntimeError((
						f"No refined gene sets found. Has refinement been run?"
					))
				except KeyError:
					raise RuntimeError((
						f"Couldn't find gene set for k = {k} "
						f"component {comp_num}. Has refinement been run?"
					))
				
				gmt_rows_l.append(k_gs.to_gmt_row())

		orig_gmt_row = self._gs.to_gmt_row()

		out_data = '\n'.join([orig_gmt_row] + gmt_rows_l)

		with open(out_path, 'w') as f:
			f.write(out_data)

	@property
	def expression(self) -> Expression:
		return self.assert_not_None(self._expr)

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
	def phenotypes(self) -> Phenotypes: 
		return self.assert_not_None(self._phens)

	@property
	def phenotype_component_ics(
		self
	) -> Dict[int, List[Dict[str, PhenotypeComponentIC]]]:
		return self._phen_comp_ics



