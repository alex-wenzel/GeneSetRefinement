"""
An abstract class for any of the types of 2-dimensional data in the 
refinement pipeline (gene-by-sample, gene-by-component, etc.)
"""

from abc import ABC, abstractmethod
from multiprocessing import Value
import pandas as pd
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Union
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

	HOW_T = Literal["any"] | Literal["all"]

	def drop_nan_from_axis(
		self,
		axis: Literal[0] | Literal[1],
		how: HOW_T
	) -> None:
		"""
		Wrapper around `pandas.DataFrame.dropna()`. `how` must be `"any"` or
		`"all"`.
		"""
		self._data = self._data.dropna(
			axis = axis,
			how = how
		)

	class NanFilterException(Exception):
		def __init__(
			self,
			data: "Data2D",
			axis: Literal["rows"] | Literal["columns"],
			how: Literal["any"] | Literal["all"]
		) -> None:
			"""
			"""
			if axis == "rows":
				item_names = data.row_title
			elif axis == "columns":
				item_names = data.col_title
			else:
				raise ValueError(  #type: ignore
					f"Expected 'rows' or 'columns' for `axis`, got {axis}."
				)

			msg = (
				f"Filtering {how} NaN {item_names} ({axis}) from "
				f"{data.data_name} removed all {item_names}."
			)

			super().__init__(msg)

	IN_DATA = TypeVar("IN_DATA", bound = "Data2D")

	@classmethod
	def subset(
		cls,
		data2d: IN_DATA, 
		row_names: List[str] = [],
		column_names: List[str] = [],
		drop_nan_rows: Optional[HOW_T] = None,
		drop_nan_columns: Optional[HOW_T] = None,
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

		`drop_nan_rows` : `str`, default `None`
			Can be `None`, `"any"` or `"all"`. Drops rows from the data 
			accordingly after the features have been subset. 

		`drop_nan_columns` : `str`, default `None`
			Can be `None`, `"any"` or `"all"`. Drops columns from the data 
			accordingly after the features have been subset. 
		"""
		ret_rows: List[str] = row_names
		ret_columns: List[str] = column_names

		if len(ret_rows) == 0:
			ret_rows = data2d.row_names

		if len(ret_columns) == 0:
			ret_columns = data2d.col_names

		subs = data2d[ret_rows, ret_columns]

		if drop_nan_rows is not None:
			subs.drop_nan_from_axis(0, drop_nan_rows)

			if subs.shape[0] == 0:
				#raise ValueError(
				#	f"No {subs.row_title} remain in {subs.data_name} after "
				#	f"dropping rows with {drop_nan_rows} NaN values."
				#)
				raise cls.NanFilterException(
					data2d, 
					"rows", 
					drop_nan_rows
				)

		if drop_nan_columns is not None:
			subs.drop_nan_from_axis(1, drop_nan_columns)

			if subs.shape[1] == 0:
				#raise ValueError(
				#	f"No {subs.col_title} remain in {subs.data_name} after "
				#	f"dropping columns with {drop_nan_columns} NaN values."
				#)
				raise cls.NanFilterException(
					data2d, 
					"columns", 
					drop_nan_columns
				)

		return subs

	T_DATA_1 = TypeVar("T_DATA_1", bound = "Data2D")
	T_DATA_2 = TypeVar("T_DATA_2", bound = "Data2D")

	@classmethod
	def subset_shared(
		cls,
		data1: T_DATA_1,
		data2: T_DATA_2,
		shared_rows: bool = False,
		shared_cols: bool = False,
		data1_drop_nan_rows: Optional[HOW_T] = None,
		data1_drop_nan_columns: Optional[HOW_T] = None,
		data2_drop_nan_rows: Optional[HOW_T] = None,
		data2_drop_nan_columns: Optional[HOW_T] = None
	) -> Tuple[T_DATA_1, T_DATA_2]:
		"""
		Takes two `Data2D` objects and subsets them to their shared rows
		and/or columns. One or both of `shared_rows` and `shared_cols` 
		must be set to `True`. Objects are returned in the order they were
		provided. 

		Parameters
		----------
		`data1` : `Data2D` subclass
			The first object to use for subsetting.

		`data2` : `Data2D` subclass
			The second object to use for subsetting.

		`shared_rows` : `bool`, default `False`
			If `True`, subset both objects to the feature names they have 
			in common. Must be `True` if `shared_cols` is `False`.

		`shared_cols` : `bool`, default `False`
			If `True`, subset both objects to the sample names they have
			in common. Must be `True` if `shared_rows` is `False`.

		See `subset()` documentation for `*_drop_nan_*` usage.

		Returns
		-------
		`Tuple` of (`type[data1]`, `type[data2]`)
			The objects in the order they were input. 
		"""
		if not (shared_rows or shared_cols):
			raise ValueError(
				"One or both of shared_rows and shared_cols must be True"
			)

		## Get rows to subset

		if shared_rows:
			data1_row_names = list(
				set(data1.row_names).intersection(data2.row_names)
			)

			if len(data1_row_names) == 0:
				raise ValueError(
					f"No shared row names between {data1.data_name} and "
					f"{data2.data_name} objects."
				)

			data2_row_names = data1_row_names
		else:
			data1_row_names = data1.row_names
			data2_row_names = data2.row_names

		## Get cols to subset

		if shared_cols:
			data1_col_names = list(
				set(data1.col_names).intersection(data2.col_names)
			)

			if len(data1_col_names) == 0:
				raise ValueError(
					f"No shared column names between {data1.data_name} and "
					f"{data2.data_name} objects."
				)

			data2_col_names = data1_col_names
		else:
			data1_col_names = data1.col_names
			data2_col_names = data2.col_names

		## Subset
		data1_subs = cls.subset(
			data1,
			row_names = data1_row_names,
			column_names = data1_col_names,
			drop_nan_rows = data1_drop_nan_rows,
			drop_nan_columns = data1_drop_nan_columns
		)

		data2_subs = cls.subset(
			data2,
			row_names = data2_row_names,
			column_names = data2_col_names,
			drop_nan_rows = data2_drop_nan_rows,
			drop_nan_columns = data2_drop_nan_columns
		)

		return (data1_subs, data2_subs)


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





