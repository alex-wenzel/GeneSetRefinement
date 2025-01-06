"""
Representation + access to dataframe(s) of phenotype data. 
"""

from gp.data import GCT
import pandas as pd
from typing import Dict, List, Optional

from .Utils import load_gct


class Phenotypes:
	"""
	Representation and access for phenotype data keyed by name of the 
	particular dataset. 
	"""
	_data: Dict[str, pd.DataFrame]

	def __init__(
		self,
		phen_dfs: Dict[str, pd.DataFrame]
	) -> None:
		"""
		Instantiation from already-loaded dataframes, write a classmethod
		to parse specific formats, e.g., `from_gct()`. 

		Parameters
		----------
		`phen_dfs` : `dict` of `str` to `pd.DataFrame`
			Maps name of phenotype table to the corresponding dataframe 
		"""
		self._data = phen_dfs

	@classmethod
	def from_gcts(
		cls,
		paths_d: Dict[str, str]
	) -> "Phenotypes":
		"""
		Given a dictionary of table names to filepaths, produce a new
		`Phenotypes` object containing each of the datasets. 

		Parameters
		----------
		`paths_d` : `dict` of `str` to `str`

		Returns
		-------
		`Phenotypes`
			A new phenotype object containing each of the tables. 
		"""
		phen_dfs = {
			table_name: load_gct(paths_d[table_name])
			for table_name in paths_d.keys()
		}

		return cls(phen_dfs)

	def get_phenotype_vector(
		self,
		table_name: str,
		phen_name: str,
		samples: Optional[List[str]] = None
	) -> pd.Series:
		"""
		Given a name of a phentoype table and name of a phenotype feature
		within the table, retrieve the values for that phenotype, limited
		to and in the same order as the list of samples, if provided. 

		Parameters
		----------
		`table_name` : `str`
			Name of a table as initially provided to instantiate the
			`Phenotypes` object. 

		`phen_name` : `str`
			The name of a phenotype (row/feature) within the table specified
			by `table_name`. 

		`samples` : `list` of `str`, default `None`
			If not `None`, a list of samples for which to retrieve phenotype
			data. Data will be returned in the same order as the samples 
			provided. 

		Returns
		-------
		`pd.Series`
			The vector corresponding to the requested phenotype. 
		"""
		try:
			phen_df = self._data[table_name]
		except KeyError:
			raise KeyError((
				f"Table '{table_name}' not found in phenotypes. "
				f"Tables available: {', '.join(self._data.keys())}."
			))

		try:
			phen_vec = pd.Series(phen_df.loc[phen_name,:])
		except KeyError:
			raise KeyError((
				f"Phenotype '{phen_name}' not found in table '{table_name}'."
			))

		if samples is None:
			return phen_vec
		else:
			return phen_vec[samples]






