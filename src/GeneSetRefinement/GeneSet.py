"""
Representation of a gene set with parsing functions from lists and GMTs
"""

import pandas as pd
from typing import Dict, List


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

	@property
	def name(self) -> str: return self._name

	@property
	def genes(self) -> List[str]: return self._genes











