"""
An abstract class for any of the types of 2-dimensional data in the 
refinement pipeline (gene-by-sample, gene-by-component, etc.)
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, TypeVar
from typing_extensions import Self


class Data2D(ABC):
	"""
	Implements a generic 2D dataset with subsetting with relaxed intersection
	constraints so pandas checks don't have to be rewritten all over the code. 
	Also allows for simple forgiveness-rather-than-permission subsetting for 
	either lists of labels or a single string label. 

	Note: Provides `row_names()` and `col_names()` properties, but implementing
	classes may add more specific property functions, e.g., `sample_names()`, as
	wrappers around these. 
	"""
	_data: pd.DataFrame

	_base_attrs: Dict[str, Any]

	def __init__(
		self,
		data: pd.DataFrame
	) -> None:
		"""
		Provide the DataFrame that the implementation of `Data2D` provides
		access to. 

		Parameters
		----------
		`data` : `pd.DataFrame`
			The two-dimensional data to provide access for. 
		"""
		self._data = data

		self._base_attrs = {}

		for key in vars(self).keys():
			self._base_attrs[key] = vars(self)[key]

	def _check_subset_input(
		self,
		subset_labels: str|List[str],
		axis_labels: List[str]
	) -> Optional[List[str]]:
		"""
		If `subset_labels` is a list, return the items that are in 
		`axis_labels`. If intersection is empty, return `None`. If 
		`subset_labels` is a `str`, return it if it is in `axis_labels`, 
		otherwise return `None`. 

		Parameters
		----------
		`subset_labels` : `str` or `list` of `str`
			The label(s) of the subset being requested. 

		`axis_labels` : `List[str]`
			The current axis labels for this object. 

		Returns
		-------
		`str`, `list` of `str`, `None`
			`subset_labels` found in `axis_labels`, see function description. 
		"""
		if isinstance(subset_labels, str):
			subset_labels_list = [subset_labels]
		else:
			subset_labels_list = subset_labels

		found_labels = [
			x for x in subset_labels_list if x in axis_labels
		]

		if len(found_labels) == 0:
			return None

		return found_labels

	def __getitem__(
		self,
		keys: Tuple[List[str] | str, List[str] | str]
	) -> Self:
		"""
		Takes two iterables of strings (list or series generally) as row names
		and column names to subset respectively and returns a new object. 

		Parameters
		----------
		"""
		rows, cols = keys

		shared_rows = self._check_subset_input(rows, self.row_names)

		if shared_rows is None:
			raise KeyError((
				f"None of requested {self.row_title} names "
				f"are in this {self.data_name}."
			))

		shared_cols = self._check_subset_input(cols, self.col_names)

		if shared_cols is None:
			raise KeyError((
				f"None of requested {self.col_title} names "
				f"are in this {self.data_name}."
			))

		filt_data: pd.DataFrame = self.data.loc[shared_rows, shared_cols]

		child_kwargs: Dict[str, Any] = {
			key.strip('_'): value
			for key, value in self._get_child_attrs().items()
		}

		return self.__class__(filt_data, **child_kwargs)

	def _check_attrs(
		self,
		attrs: List[str]
	) -> None:
		"""
		Checks if this object has the specified attributes. If not, raises
		an `AttributeError` that advises the user to make sure the implementing
		class calls `super().__init__()`. 
		"""
		missing_attrs: List[str] = [
			a for a in attrs if not hasattr(self, a)
		]

		if len(missing_attrs) == 0:
			return

		cls_name: str = self.__class__.__name__

		raise AttributeError((
			f"Missing attribute(s) {', '.join(missing_attrs)}. "
			f"Does {cls_name}.__init__() call super().__init__()?"
		))

	def _get_child_attrs(self) -> Dict[str, Any]:
		"""
		Returns any attribute names and values that are added by a child of
		`Data2D`. 

		Returns
		-------
		`dict` of `str` to any type
		"""
		return {
			key: vars(self)[key]
			for key in vars(self).keys()
			if key not in self._base_attrs
		}

	IN_DATA = TypeVar("IN_DATA", bound = "Data2D")
	@classmethod
	def subset(
		cls,
		data2d: IN_DATA, 
		row_names: List[str] = [],
		column_names: List[str] = []
	) -> IN_DATA:
		"""
		Returns a new Expression object subset to the given gene and 
		column names. 

		Parameters
		----------
		`data2d` : `Data2D` 
			An instance of any subclass implementing `Data2D`. 

		`row_names` : `Iterable` of `str`
			The rows to include in the subset. If empty, include all rows.

		`column_names` : `Iterable` of `str`
			The columns to include in the subset. If empty, include
			all columns. 
		"""
		ret_rows: List[str] = row_names
		ret_columns: List[str] = column_names

		if len(ret_rows) == 0:
			ret_rows = data2d.row_names

		if len(ret_columns) == 0:
			ret_columns = data2d.col_names

		return data2d[ret_rows, ret_columns]

	def squeeze(
		self
	) -> List[float]:
		"""
		Checks that at least one dimension has only one entry and 
		returns that as a list of floats. 
		"""
		if self.shape[0] == 1:
			return self.data.iloc[0,:].to_list()

		elif self.shape[1] == 1:
			return self.data.iloc[:,0].to_list()

		else:
			raise ValueError(
				f"Cannot call squeeze() for data of shape {self.shape}."
			)

	@property
	@abstractmethod
	def data_name(self) -> str:
		pass

	@property
	@abstractmethod
	def row_title(self) -> str:
		pass

	@property
	@abstractmethod
	def col_title(self) -> str:
		pass

	@property
	def row_names(self) -> List[str]:
		self._check_attrs(["_data"])
		return self._data.index.to_list()

	@property
	def col_names(self) -> List[str]: 
		self._check_attrs(["_data"])
		return self._data.columns.to_list()

	@property
	def data(self) -> pd.DataFrame:
		self._check_attrs(["_data"])
		return self._data

	@property
	def shape(self) -> Tuple[int, int]:
		self._check_attrs(["_data"])
		return self._data.shape





