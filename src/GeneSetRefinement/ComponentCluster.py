import numpy as np
import pandas as pd
from typing import List, TYPE_CHECKING

from .Data2D import Data2DView
from .GeneComponentIC import CombinedGeneComponentIC
from .GeneSet import GeneSet
from .RefinementObject import RefinementObject

if TYPE_CHECKING:
	from .Refinement import Refinement


class ComponentCluster(RefinementObject):
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