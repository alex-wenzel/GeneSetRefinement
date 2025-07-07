"""
Representation of an expression matrix with subsetting functions for
refinement. 
"""

from gp.data import GCT
import pandas as pd

from .Data2D import Data2D
from .Utils import load_gct


class Expression(Data2D):
	"""
	Expression matrix representation. Loads from a GCT. 
	"""
	def __init__(
		self,
		data: pd.DataFrame,
	) -> None:
		super().__init__(data)

	@classmethod
	def from_gct(
		cls,
		gct_path: str,
		min_counts: int = 5
	) -> "Expression":
		"""
		Returns an `Expression` object representing the data at `gct_path`. 

		Parameters
		----------
		`gct_path` : `str` 
			A path to a GCT file. 

		`min_counts` : `int`
			The minimum number of total counts for a gene across all samples
			for the gene to be retained in the expression matrix after loading. 

		Returns
		-------
		`Expression`
			A representation of the GCT data filtered to `min_counts`. 
		"""
		data = load_gct(gct_path)

		data = data.loc[data.sum(axis = 1) > min_counts, :]

		return cls(data)

	@property
	def data_name(self) -> str: return "gene expression matrix"

	@property 
	def row_title(self) -> str: return "gene"

	@property
	def col_title(self) -> str: return "sample"

	@property
	def n_genes(self) -> int:
		return self._data.shape[0]

	@property
	def n_samples(self) -> int:
		return self._data.shape[1]

	def normalize(self) -> None:
		"""
		10000 * (x - min(x)) / (max(x) - min(x))
		"""
		self._data = self._data.rank(
			axis = 0,
			method = "dense",
			numeric_only = False,
			na_option = "keep",
			ascending = True,
			pct = False
		)

		expr_min = self._data.min()
		expr_max = self._data.max()

		min_max_diff = expr_max - expr_min

		self._data = 10000 * (self._data - expr_min) / min_max_diff










