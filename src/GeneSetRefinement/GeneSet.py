"""
Representation of a gene set with parsing functions from lists and GMTs
"""

import json
from pathlib import Path
import os
import requests
from typing import TypedDict, Dict, List, Optional


class MSigDBJSON(TypedDict):
	collection: str
	systematicName: str
	pmid: str
	exactSource: str
	externalDetailsURL: str
	msigdbURL: str
	geneSymbols: List[str]
	filteredBySimilarity: List[str]
	externalNamesForSimilarTerms: List[str]

class MSigDBQuery(TypedDict):
	release: str
	collection: str
	gene_set_name: str

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

			gene_set_genes = [gene for gene in split[2:] if gene]

			gs_d[gene_set_name] = gene_set_genes

	return gs_d


class GeneSet:
	_name: str
	_description: Optional[str]
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
		gene_set_genes: List[str],
		description: Optional[str] = None
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
		self._name = gene_set_name
		self._genes = gene_set_genes
		self._description = description

	@classmethod
	def all_from_gmt(
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
	
	@classmethod
	def from_gmt(
		cls,
		gmt_path: str,
		gene_set_name: str
	) -> "GeneSet":
		"""
		"""
		full_gmt_d = cls.all_from_gmt(gmt_path)

		return full_gmt_d[gene_set_name]
	
	@classmethod
	def _get_gs_from_msigdb_json(
		cls,
		msigdb_dict_or_json: str | Dict[str, MSigDBJSON],
		gene_set_name: str
	) -> "GeneSet":
		"""
		"""
		if isinstance(msigdb_dict_or_json, str):

			with open(msigdb_dict_or_json, 'r') as f:
				msigdb_d: Dict[str, MSigDBJSON] = json.load(f)

		else:
			msigdb_d = msigdb_dict_or_json

		return cls(
			gene_set_name,
			msigdb_d[gene_set_name]["geneSymbols"]
		)
	
	@classmethod
	def from_msigdb(
		cls,
		msigdb_query: MSigDBQuery,
		msigdb_url: str = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/",
		cache_dir: str = ".msigdb/",
		use_cache: bool = True
	) -> "GeneSet":
		"""
		"""
		collection = msigdb_query["collection"]
		release = msigdb_query["release"]
		gene_set_name = msigdb_query["gene_set_name"]

		msigdb_json_name = f"{collection}.v{release}.json"

		if use_cache:
			if not os.path.exists(cache_dir):
				os.makedirs(cache_dir)

			try_msigdb_json_path = Path(cache_dir) / f"{msigdb_json_name}"

			try:
				return cls._get_gs_from_msigdb_json(
					str(try_msigdb_json_path),
					gene_set_name
				)
			except FileNotFoundError:
				pass

		full_msigdb_url = msigdb_url

		if full_msigdb_url[-1] != "/":
			full_msigdb_url = f"{full_msigdb_url}/"

		full_msigdb_url = f"{full_msigdb_url}{release}/{msigdb_json_name}"

		req = requests.get(full_msigdb_url)

		if req.status_code != 200:
			raise requests.RequestException(
				f"Request for '{req.url}' return code {req.status_code}."
			)
		
		msigdb_d: Dict[str, MSigDBJSON] = req.json()

		if use_cache:
			with open(try_msigdb_json_path, 'w') as f:
				json.dump(msigdb_d, f)

		return cls._get_gs_from_msigdb_json(msigdb_d, gene_set_name)
		
	
	def to_gmt_row(
		self
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
		if self._description is None:
			description = self.name
		else:
			description = self._description

		return '\t'.join([self.name, description] + self.genes)

	@property
	def name(self) -> str: return self._name

	@property
	def genes(self) -> List[str]: return self._genes











