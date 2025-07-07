import numpy as np
import pandas as pd
from typing import List, TYPE_CHECKING

from .Data2D import Data2D, Data2DView
from .GeneSet import GeneSet
from .Phenotypes import Phenotype
from .Utils import compute_information_coefficient, ssGSEAResult, Log

if TYPE_CHECKING:
	from .Refinement import Refinement

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
				list(map(
					str, 
					(
						ssgsea_data.data.loc[gene_set_name,:]
						.squeeze().to_list()[:10]
					)
				)) 
			)

			phen_preview = ','.join(
				list(map(
					str, 
					(
						phen_vec.data.loc[phen_feature_name,:]
						.squeeze().to_list()[:10]
					)
				)) 
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
		if refinement.phenotypes is None:
			raise ValueError(
				f"Cannot compute phenotype associations for '{phen_name}', "
				f"data is missing (None)."
			)

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
					phen_vec = phen_data.subset(
						row_names = [phen_feature_name],
						column_names = one_ssgsea_res.col_names
					)

				except Data2D.NanFilterException:
					ic_array[x, y] = np.nan
					continue

				one_ssgsea_res = one_ssgsea_res.subset(
					column_names = phen_vec.col_names
				)

				try:
					ic = compute_information_coefficient(
						one_ssgsea_res.squeeze(),
						phen_vec.squeeze(),
						raise_if_failed = False
					)
					
				except Exception as e:
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