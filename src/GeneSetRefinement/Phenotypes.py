"""
Representation + access to dataframe(s) of phenotype data. 
"""

from gp.data import GCT
import pandas as pd
from typing import Dict, List, Optional

from .Data2D import Data2D
from .Utils import load_gct


class Phenotype(Data2D):
	_phen_name: str

	def __init__(
		self,
		data: pd.DataFrame,
		phen_name: str
	) -> None:
		"""
		"""
		super().__init__(data)
		self._phen_name = phen_name

	@property
	def data_name(self) -> str: return "phenotype table"

	@property
	def row_title(self) -> str: return "phenotype feature"

	@property
	def col_title(self) -> str: return "phenotype sample"

	@property
	def phenotype_name(self) -> str: return self._phen_name


class Phenotypes:
	"""
	Representation and access for phenotype data keyed by name of the 
	particular dataset. 
	"""
	_data: Dict[str, Phenotype]
	_phen_table_names: List[str]

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
		self._data = {}

		for name, df in phen_dfs.items():
			self._data[name] = Phenotype(df, name)

		## freeze order of names for consistent access
		self._phen_table_names = list(self._data.keys())


	def __getitem__(self, key: str) -> Phenotype:
		return self._data[key]

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
			phen_vec = pd.Series(phen_df.data.loc[phen_name,:])
		except KeyError:
			raise KeyError((
				f"Phenotype '{phen_name}' not found in table '{table_name}'."
			))

		if samples is None:
			return phen_vec
		else:
			return phen_vec[samples]

	@property
	def phenotype_table_names(self) -> List[str]: return self._phen_table_names



