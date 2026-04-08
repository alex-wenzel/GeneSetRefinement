from multiprocessing.synchronize import Lock as TLock
from multiprocessing import Queue
import numpy as np
import pandas as pd
from typing import List, Optional, TYPE_CHECKING

from .Data2D import Data2DView
from .Expression import Expression
from .GeneComponentIC import CombinedGeneComponentIC, GeneComponentIC
from .GeneSet import GeneSet
from .NMFResult import NMFResult, NMF_W_Matrix, NMF_H_Matrix
from .RefinementObject import RefinementObject
from .Utils import Log

if TYPE_CHECKING:
	from .Refinement import Refinement


class InnerIteration(RefinementObject):
	## Inputs
	_train_expr: Optional[Data2DView[Expression]]
	_train_samples: Optional[List[str]]
	_gs: GeneSet
	_i: int ## external outer index
	_j: int ## external inner index
	_k: int
	_seed: int
	_log: Log
	_max_tries: int

	## Intermediates + results
	_rng: np.random.Generator
	_gen_expr: Optional[Data2DView[Expression]]
	_gen_samples: Optional[List[str]]
	_A: Optional[Data2DView[Expression]]
	_A_samples: Optional[List[str]]
	_nmf_result: NMFResult
	_gene_comp_ic: GeneComponentIC

	def __init__(
		self,
		training_expression: Data2DView[Expression],
		input_gene_set: GeneSet,
		i: int,
		j: int,
		k: int,
		seed: int,
		log: Log,
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
		self._train_samples = training_expression.col_names
		self._gs = input_gene_set
		self._i = i
		self._j = j
		self._k = k
		self._seed = seed
		self._rng = np.random.default_rng(seed)
		self._log = log
		self._max_tries = max_downsample_tries

		#self._pref = f"Inner iteration {self._i}.{self._j} -"
		self._pref = f"(k={self._k}|i={self._i}|j={self._j}) -"
	
	def run(
		self,
		log_q: Queue
	) -> None:
		"""
		Driver function. 
		"""
		#self._log(f"{self._pref} Creating generating set...")
		log_q.put(f"{self._pref} Creating generating set...")
		self._train_expr = self.assert_not_None(self._train_expr)
		self._gen_expr = self._train_expr.subset_random_samples(
			0.5,
			self._rng,
			return_both = False,
			max_tries = self._max_tries
		)
		self._gen_samples = self._gen_expr.col_names
		#self._log(f"{self._pref} done creating generating set.")
		log_q.put(f"{self._pref} done creating generating set.")

		self._A = self._get_A()
		self._A_samples = self._A.col_names

		self._A = self.assert_not_None(self._A)

		#self._log(f"{self._pref} Computing NMF...")
		log_q.put(f"{self._pref} Computing NMF...")
		self._nmf_result = NMFResult(
			self._A,
			self._k,
			self._rng
		)
		self._nmf_result.run()
		#self._log(f"{self._pref} done running NMF.")
		log_q.put(f"{self._pref} done running NMF.")

		#self._log(f"{self._pref} computing gene/component IC...")
		log_q.put(f"{self._pref} computing gene/component IC...")
		self._gene_comp_ic = GeneComponentIC(
			self._nmf_result,
			self._i,
			self._j
		)
		#self._log(f"{self._pref} done computing gene/component IC.")
		log_q.put(f"{self._pref} done computing gene/component IC.")

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
		self._gen_expr = self.assert_not_None(self._gen_expr)

		shared_genes = [
			gene for gene in self._gs.genes
			if gene in self._gen_expr.row_names
		]

		return self._gen_expr.subset(row_names = shared_genes)


	@property
	def k(self) -> int: return self._k

	@property
	def A(self) -> Data2DView[Expression]: 
		return self.assert_not_None(self._A)

	@property
	def W(self) -> NMF_W_Matrix: return self._nmf_result.W

	@property
	def H(self) -> NMF_H_Matrix: return self._nmf_result.H

	@property
	def gene_component_IC(self) -> GeneComponentIC: return self._gene_comp_ic

	@property
	def training_expression(self) -> Data2DView[Expression]: 
		return self.assert_not_None(self._train_expr)

	@property
	def generating_expression(self) -> Data2DView[Expression]:
		return self.assert_not_None(self._gen_expr)

