"""
Representation of a gene set with parsing functions from lists and GMTs
"""

import pandas as pd
from typing import Dict, List, Optional


def read_gmt(
	gmt_path: str
) -> Dict[str, List[str]]:
	"""
	Based on ccalnoir: 
	https://github.com/alex-wenzel/ccal-noir/blob/master/ccalnoir/elemental.py

	Loads a GMT and returns it as a dictionary of gene lists keyed by
	gene set name. 

	Parameters
	----------
	`gmt_path` : `str`
		Path to a GMT file. 

	Returns
	-------
	`dict` of `str` to `list` of `str`
		Gene sets keyed by gene set name. 
	"""
	gs_d: Dict[str, List[str]] = {}

	with open(gmt_path, 'r') as gmt_file:
		for line in gmt_file:
			split = line.strip().split(sep = '\t')

			gene_set_name = split[0]

			gene_set_genes = [
				gene for gene in set(split[2:]) if gene
			]

			gs_d[gene_set_name] = gene_set_genes

	return gs_d


class GeneSet:
	_name: str
	_genes: List[str]

	class EmptyGeneSetException(Exception):
		def __init__(
			self,
			gene_set_name: str
		) -> None:
			"""
			"""
			super().__init__(
				f"Tried to instantiate gene set '{gene_set_name}' with 0 genes."
			)

	def __init__(
		self,
		gene_set_name: str,
		gene_set_genes: List[str]
	) -> None:
		"""
		Given a name and list of genes, instantiate a `GeneSet` object. 

		Parameters
		----------
		`name` : `str`
			The name of the gene set. 

		`genes` : `List[str]`
			The gene set genes. 
		"""
		#if len(gene_set_genes) == 0:
		#	raise self.EmptyGeneSetException(gene_set_name)

		self._name = gene_set_name
		self._genes = gene_set_genes

	@classmethod
	def from_gmt(
		cls,
		gmt_path: str
	) -> Dict[str, "GeneSet"]:
		"""
		"""
		gs_d = read_gmt(gmt_path)

		return {
			gs_name: cls(gs_name, gs_d[gs_name])
			for gs_name in gs_d.keys()
		}
	
	def to_gmt_row(
		self,
		description: Optional[str] = None
	) -> str:
		"""
		Returns a string representing a single row of a GMT file. 
		This DOES NOT include a newline. 

		Parameters
		----------
		`description` : `str`
			Value to use in the second column of the GMT. If not provided,
			the gene set name will be used, i.e., the first and second column
			will have the same values. 

		Returns
		-------
		`str`
			The full row of the gene set's representation in GMT format, 
			without a newline. 
		"""
		if description is None:
			description = self.name

		return '\t'.join([self.name, description] + self.genes)

	@property
	def name(self) -> str: return self._name

	@property
	def genes(self) -> List[str]: return self._genes











