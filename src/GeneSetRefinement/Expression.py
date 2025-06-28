"""
Representation of an expression matrix with subsetting functions for
refinement. 
"""

from gp.data import GCT
import numpy as np
import pandas as pd
from typing import List, Literal, Optional, Tuple, TypeVar, overload, Union
from typing_extensions import Self

from .Data2D import Data2D, Data2DView
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

	"""
	@property
	def gene_names(self) -> List[str]:
		return self.row_names

	@property
	def sample_names(self) -> List[str]:
		return self.col_names
	"""

	@property
	def n_genes(self) -> int:
		return self._data.shape[0]

	@property
	def n_samples(self) -> int:
		return self._data.shape[1]

	#@property
	#def min_counts(self) -> int:
	#	return self._min_counts

	"""
	@overload
	def subset_random_samples(
		self,
		frac: float,
		rng: np.random.Generator,
		return_both: Literal[False],
		no_zero_rows: bool = True,
		max_tries: int = 100
	) -> "Data2DView[Self]": ...
	@overload
	def subset_random_samples(
		self,
		frac: float,
		rng: np.random.Generator,
		return_both: Literal[True],
		no_zero_rows: bool = True,
		max_tries: int = 100
	) -> "Tuple[Data2DView[Self], Data2DView[Self]]": ...
	def subset_random_samples(
		self,
		frac: float,
		rng: np.random.Generator,
		return_both: bool = False,
		no_zero_rows: bool = True,
		max_tries: int = 100
	) -> "Data2DView[Self] | Tuple[Data2DView[Self], Data2DView[Self]]":
		###
		Returns either one or two `Expression` objects containing randomly
		subset samples. One object will have (`frac` * #columns) samples, 
		the other optional object will have (`1 - frac` * #columns) samples. 

		Parameters
		----------
		`frac` : `float`
			A fraction of samples to downsample. Must be between 0.0 and 1.0

		`rng` : `np.random.Generator`
			A pre-instantiated RNG object, see Numpy documentation for details.

		`return_both` : `bool`, default `False`
			If `True`, return two matrices with `frac` of original samples and 
			(`1 - frac`) of original samples respectively. If `False`, returns
			only the first matrix with `frac` of original samples. 

		`no_zero_rows` : `bool`, default `True`
			If `True`, will not accept a random subset that contains rows that
			are all zero. Will try `max_tries` times to achieve this before 
			throwing an error. If `False`, this condition is not enforced and
			the first random subset will be returned.

		`max_tries` : `int`, default `100`
			If `no_zero_rows` is `True`, the number of attempts at generating
			a compliant random subset until an error is thrown. This parameter
			is not used if `no_zero_rows` is `False`. 
		###
		if (frac < 0.0) or (frac > 1.0):
			raise ValueError(
				"Fraction of Expression sampled must be between 0 and 1"
			)

		## Define loop stop checks
		## Convention: 'keep' is first matrix, 'disc' (discard) is second matrix
		keep_has_zero_row = True
		disc_has_zero_row = True
		tries = 0

		## Define sampling checks
		n_to_choose = int(self.n_samples * frac)

		## Sampling loop
		while True:
			## Choose columns for 'keep' matrix
			keep_cols = rng.choice(
				self.col_names,
				size = n_to_choose,
				replace = False
			).tolist()

			## Build keep and discard matrices
			disc_cols = [
				col for col in self.col_names
				if col not in keep_cols
			]

			#keep = Data2D.subset(self, self.gene_names, keep_cols)
			#disc = Data2D.subset(self, self.gene_names, disc_cols)

			keep = self.subset(self.row_names, keep_cols)
			disc = self.subset(self.row_names, disc_cols)

			if no_zero_rows:
				keep_has_zero_row = keep.data2d.has_zero_row()
				disc_has_zero_row = disc.data2d.has_zero_row()

				if (not keep_has_zero_row) and (not disc_has_zero_row):
					break

				tries += 1

				if tries == max_tries:
					raise RuntimeError((
						f"{max_tries} unsuccessful attempts to subset matrix "
						f"to fraction {frac} of {self.n_samples} samples."
					))

			else:
				break

		if return_both: 
			return keep, disc
		else:
			return keep
	"""

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










