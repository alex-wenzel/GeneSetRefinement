import numpy as np
import pandas as pd
from typing import List, TYPE_CHECKING

from .Data2D import Data2D
from .NMFResult import NMFResult
from .Utils import compute_information_coefficient

if TYPE_CHECKING:
	from .Refinement import Refinement


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