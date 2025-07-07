from abc import abstractmethod
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from typing import List, Optional
from typing_extensions import Self

from .Data2D import Data2D, Data2DView
from .Expression import Expression
from .RefinementObject import RefinementObject

class NMFResultMatrix(Data2D):
	@classmethod
	def _np2df(
		cls,
		arr: np.ndarray,
		index: Optional[List[str]] = None,
		columns: Optional[List[str]] = None
	) -> pd.DataFrame:
		df_index = index
		df_columns = columns

		if df_index is None:
			df_index = [
				str(i) for i in range(arr.shape[0])
			]

		if df_columns is None:
			df_columns = [
				str(i) for i in range(arr.shape[1])
			]

		df = pd.DataFrame(arr, index = df_index, columns = df_columns)

		return df

	@classmethod
	@abstractmethod
	def from_array(
		cls,
		arr: np.ndarray,
		index: Optional[List[str]] = None,
		columns: Optional[List[str]] = None
	) -> Self:
		pass


class NMF_W_Matrix(NMFResultMatrix):
	@classmethod
	def from_array(
		cls,
		arr: np.ndarray,
		index: Optional[List[str]] = None,
		columns: Optional[List[str]] = None
	) -> "NMF_W_Matrix":
		"""
		"""
		df = cls._np2df(
			arr, 
			index = index, 
			columns = columns
		)

		return NMF_W_Matrix(df)

	@property
	def data_name(self) -> str: return "W matrix"

	@property
	def row_title(self) -> str: return "gene"

	@property
	def col_title(self) -> str: return "component"


class NMF_H_Matrix(NMFResultMatrix):
	@classmethod
	def from_array(
		cls,
		arr: np.ndarray,
		index: Optional[List[str]] = None,
		columns: Optional[List[str]] = None
	) -> "NMF_H_Matrix":
		"""
		"""
		df = cls._np2df(
			arr,
			index = index,
			columns = columns
		)

		return NMF_H_Matrix(df)

	@property
	def data_name(self) -> str: return "H matrix"

	@property
	def row_title(self) -> str: return "component"

	@property
	def col_title(self) -> str: return "sample"


class NMFResult(RefinementObject):
	_A: Optional[Data2DView[Expression]]
	_nmf: NMF
	_W: NMF_W_Matrix
	_H: NMF_H_Matrix

	def __init__(
		self,
		A: Data2DView[Expression],
		k: int,
		rng: np.random.Generator
	) -> None:
		"""
		Instantiate an instance of a single NMF solution for one inner
		iteration. 

		Parameters
		----------
		`A` : `Data2DView[Expression]`
			The `A` matrix to decompose, a subset of the original expression. 

		`k` : `int`
			The value of `k` (number of components) to use. 

		`rng` : `np.random.Generator`
			The random state as passed to the inner iteration. 
		"""
		self._A = A

		self._nmf = NMF(
			n_components = k,
			init = "random",
			solver = "mu",
			beta_loss = "kullback-leibler",
			max_iter = 2000,
			random_state = rng.integers(
				low = 1,
				high = np.iinfo(np.int32).max
			)
		)

	def run(self) -> None:
		"""
		Computes and stores the W and H matrices. 
		"""
		self._A = self.assert_not_None(self._A)

		self._W = NMF_W_Matrix.from_array(
			self._nmf.fit_transform(self._A.array),
			index = self.A.row_names
		)

		self._H = NMF_H_Matrix.from_array(
			self._nmf.components_,
			columns = self.A.col_names
		)

	@property
	def A(self) -> Data2DView[Expression]: 
		return self.assert_not_None(self._A)

	@property
	def W(self) -> NMF_W_Matrix: return self._W

	@property
	def H(self) -> NMF_H_Matrix: return self._H

	@property
	def k(self) -> int: return int(self._nmf.n_components_)