from abc import ABCMeta, abstractmethod
from multiprocessing import Value
import pandas as pd
import numpy as np
from typing import (
	Any, Dict, Generic, List, Literal, Optional, overload, Tuple, Type, 
	TypeVar
)
from typing_extensions import Self

class Data2DAbs(metaclass=ABCMeta):
	_base_attrs: Dict[str, Any]

	def __init__(self) -> None:
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

	def _get_row_inds(
		self,
		row_names: List[str]
	) -> List[int]:
		all_row_names = self.row_names

		#return [
		#	all_row_names.index(rn) for rn in row_names
		#]

		row_inds: List[int] = []

		for rn in row_names:
			try:
				row_inds.append(all_row_names.index(rn))
			except ValueError:
				continue

		return row_inds

	def _get_col_inds(
		self,
		column_names: List[str]
	) -> List[int]:
		all_col_names = self.col_names

		#return [
		#	all_col_names.index(cn) for cn in column_names
		#]

		col_inds: List[int] = []

		for cn in column_names:
			try:
				col_inds.append(all_col_names.index(cn))
			except ValueError:
				continue

		return col_inds


	class NanFilterException(Exception):
		def __init__(
			self,
			data: "Data2DAbs",
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

	@abstractmethod
	def __getitem__(
		self,
		keys: Tuple[List[str] | str, List[str] | str]
	) -> Self:
		pass

	HOW_T = Literal["any"] | Literal["all"]

	@abstractmethod
	def subset(
		self,
		row_names: List[str] = [],
		column_names: List[str] = [],
		drop_nan_rows: Optional[HOW_T] = None,
		drop_nan_columns: Optional[HOW_T] = None,
	) ->  "Data2DView":
		...

	@abstractmethod
	def drop_nan_from_axis(
		self,
		axis: Literal[0] | Literal[1],
		how: HOW_T
	) -> None:
		pass

	OTHER_T = TypeVar("OTHER_T", bound = "Data2D")

	@abstractmethod
	def subset_shared(
		self,
		other: "OTHER_T | Data2DView[OTHER_T]",
		shared_rows: bool = False,
		shared_cols: bool = False,
		self_drop_nan_rows: Optional[HOW_T] = None,
		self_drop_nan_columns: Optional[HOW_T] = None,
		other_drop_nan_rows: Optional[HOW_T] = None,
		other_drop_nan_columns: Optional[HOW_T] = None
	) -> "Tuple[Data2DView[OTHER_T], Data2DView[OTHER_T]]":
		pass

	@overload
	def subset_random_samples(
		self,
		frac: float,
		rng: np.random.Generator,
		return_both: Literal[False],
		no_zero_rows: bool = True,
		max_tries: int = 100
	) -> "Data2DView": 
		...
	@overload
	def subset_random_samples(
		self,
		frac: float,
		rng: np.random.Generator,
		return_both: Literal[True],
		no_zero_rows: bool = True,
		max_tries: int = 100
	) -> "Tuple[Data2DView, Data2DView]":
		...
	@abstractmethod
	def subset_random_samples(
		self,
		frac: float,
		rng: np.random.Generator,
		return_both: bool = False,
		no_zero_rows: bool = True,
		max_tries: int = 100
	) -> "Data2DView | Tuple[Data2DView, Data2DView]":
		pass

	@abstractmethod
	def has_zero_row(self) -> bool:
		pass

	@abstractmethod
	def squeeze(
		self
	) -> List[float]:
		pass

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
	@abstractmethod
	def row_names(self) -> List[str]:
		pass

	@property
	@abstractmethod
	def col_names(self) -> List[str]:
		pass

	@property
	@abstractmethod
	def data(self) -> pd.DataFrame:
		pass

	@property
	@abstractmethod
	def shape(self) -> Tuple[int, int]:
		pass

	@property
	@abstractmethod
	def array(self) -> np.ndarray:
		pass


class Data2D(Data2DAbs, metaclass=ABCMeta):
	_data: pd.DataFrame

	def __init__(
		self,
		data: pd.DataFrame
	) -> None:
		"""
		"""
		super().__init__()
		self._data = data

	def __getitem__(
		self, 
		keys: Tuple[List[str] | str, List[str] | str]
	) -> Self:
		"""
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

		#child_kwargs: Dict[str, Any] = {
		#	key.strip('_'): value
		#	for key, value in self._get_child_attrs().items()
		#}

		#return self.__class__(filt_data, **child_kwargs)

		return self.__class__(filt_data)

	HOW_T = Literal["any"] | Literal["all"]

	def subset(
		self,
		row_names: List[str] = [],
		column_names: List[str] = [],
		drop_nan_rows: Optional[HOW_T] = None,
		drop_nan_columns: Optional[HOW_T] = None,
	) ->  "Data2DView[Self]":
		"""
		"""
		ret_rows: List[str] = row_names
		ret_columns: List[str] = column_names

		if len(ret_rows) == 0:
			ret_rows = self.row_names

		if len(ret_columns) == 0:
			ret_columns = self.col_names

		## Not saved, to trigger a KeyError if any missing names
		try:
			self.data.loc[ret_rows, ret_columns]
		except KeyError:
			raise KeyError((
				f"None of requested {self.row_title} names "
				f"are in this {self.data_name}."
			))

		subs = Data2DView(
			self, 
			self._get_row_inds(ret_rows), 
			self._get_col_inds(ret_columns)
		)

		if drop_nan_rows is not None:
			subs.drop_nan_from_axis(1, drop_nan_rows)

			if subs.shape[0] == 0:
				raise self.NanFilterException(self, "rows", drop_nan_rows)

		if drop_nan_columns is not None:
			subs.drop_nan_from_axis(0, drop_nan_columns)

			if subs.shape[1] == 0:
				raise self.NanFilterException(self, "columns", drop_nan_columns)

		return subs

	def drop_nan_from_axis(
		self,
		axis: Literal[0] | Literal[1],
		how: HOW_T
	) -> None:
		"""
		"""
		self._data = self._data.dropna(
			axis = axis,
			how = how
		)

	OTHER_T = TypeVar("OTHER_T", bound = "Data2D")

	def subset_shared(
		self,
		other: "OTHER_T | Data2DView[OTHER_T]",
		shared_rows: bool = False,
		shared_cols: bool = False,
		self_drop_nan_rows: Optional[HOW_T] = None,
		self_drop_nan_columns: Optional[HOW_T] = None,
		other_drop_nan_rows: Optional[HOW_T] = None,
		other_drop_nan_columns: Optional[HOW_T] = None
	) -> "Tuple[Data2DView[Self], Data2DView[OTHER_T]]":
		"""
		"""
		if not (shared_rows or shared_cols):
			raise ValueError(
				"One or both of shared_rows and shared_cols must be True"
			)

		## Get rows to subset

		if shared_rows:
			self_row_names = list(
				set(self.row_names).intersection(other.row_names)
			)

			if len(self_row_names) == 0:
				raise ValueError(
					f"No shared row names between {self.data_name} and "
					f"{other.data_name} objects."
				)

			other_row_names = self_row_names

		else:
			self_row_names = self.row_names
			other_row_names = other.row_names

		## Get columns to subset

		if shared_cols:
			self_col_names = list(
				set(self.col_names).intersection(other.col_names)
			)

			if len(self_col_names) == 0:
				raise ValueError(
					f"No shared column names between {self.data_name} and "
					f"{other.data_name} objects."
				)

			other_col_names = self_col_names

		else:
			self_col_names = self.col_names
			other_col_names = other.col_names

		self_subs = self.subset(
			row_names = self_row_names,
			column_names = self_col_names,
			drop_nan_rows = self_drop_nan_rows,
			drop_nan_columns = self_drop_nan_columns
		)

		other_subs = other.subset(
			row_names = other_row_names,
			column_names = other_col_names,
			drop_nan_rows = other_drop_nan_rows,
			drop_nan_columns = other_drop_nan_columns
		)

		return (self_subs, other_subs)

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
		if (frac < 0.0) or (frac > 1.0):
			raise ValueError(
				"Fraction of Expression sampled must be between 0 and 1"
			)

		## Define loop stop checks
		## Convention: 'keep' is first matrix, 'disc' (discard) is second matrix

		keep_has_zero_rows = True
		disc_has_zero_rows = True
		tries = 0

		## Define sampling checks
		n_to_choose = int(self.shape[1] * frac)

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
						f"to fraction {frac} of {self.shape[1]} samples."
					))

			else:
				break

		if return_both: 
			return keep, disc
		else:
			return keep


	def squeeze(
		self
	) -> List[float]:
		"""
		"""
		if self.shape[0] == 1:
			return self.data.iloc[0,:].to_list()

		elif self.shape[1] == 1:
			return self.data.iloc[:,0].to_list()

		else:
			raise ValueError(
				f"Cannot call squeeze() for data of shape {self.shape}."
			)

	def has_zero_row(self) -> bool:
		return min(self._data.sum(axis = 1)) == 0

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

	@property
	def array(self) -> np.ndarray:
		return self._data.to_numpy()


REAL_T = TypeVar("REAL_T", bound = "Data2D")

class Data2DView(Data2DAbs, Generic[REAL_T]):
	_data2d: REAL_T
	_row_inds: List[int]
	_col_inds: List[int]

	def __init__(
		self,
		data2d: REAL_T,
		row_inds: List[int],
		col_inds: List[int]
	) -> None:
		"""
		"""
		super().__init__()
		self._data2d = data2d
		self._row_inds = row_inds
		self._col_inds = col_inds

	def __getitem__(
		self,
		keys: Tuple[List[str] | str, List[str] | str]
	) -> "Data2DView[REAL_T]":
		"""
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

		return Data2DView(
			self._data2d,
			self._get_row_inds(shared_rows),
			self._get_col_inds(shared_cols)
		)

	HOW_T = Literal["any"] | Literal["all"]

	def subset(
		self,
		row_names: List[str] = [],
		column_names: List[str] = [],
		drop_nan_rows: Optional[HOW_T] = None,
		drop_nan_columns: Optional[HOW_T] = None,
	) -> "Data2DView[REAL_T]":
		"""
		"""
		ret_rows: List[str] = row_names
		ret_columns: List[str] = column_names

		if len(ret_rows) == 0:
			ret_rows = self.row_names

		if len(ret_columns) == 0:
			ret_columns = self.col_names

		## Not saved, to trigger a KeyError if any missing names
		try:
			self._data2d.data.loc[ret_rows, ret_columns]
		except KeyError:
			raise KeyError((
				f"None of requested {self.row_title} names "
				f"are in this {self.data_name}."
			))

		subs = Data2DView(
			self._data2d, 
			self._get_row_inds(ret_rows), 
			self._get_col_inds(ret_columns)
		)

		if drop_nan_rows is not None:
			subs.drop_nan_from_axis(0, drop_nan_rows)

			if subs.shape[0] == 0:
				raise self.NanFilterException(self, "rows", drop_nan_rows)

		if drop_nan_columns is not None:
			subs.drop_nan_from_axis(1, drop_nan_columns)

			if subs.shape[1] == 0:
				raise self.NanFilterException(self, "columns", drop_nan_columns)

		return subs

	def drop_nan_from_axis(
		self,
		axis: Literal[0] | Literal[1],
		how: HOW_T
	) -> None:
		"""
		"""
		if how == "any":
			all_check =  ~ (
				self._data2d.data
				.isnull()
				.any(axis = axis)
				.reset_index(drop = True)
			)

		elif how == "all":
			all_check = ~ (
				self._data2d.data
				.isnull()
				.all(axis = axis)
				.reset_index(drop = True)
			)

		else:
			raise ValueError(
				f"how must be 'any' or 'all', got {how}"
			)

		if axis == 0:
			self._col_inds = list(
				set(self._col_inds)
				.intersection(all_check[all_check].index.to_list())
			)

		elif axis == 1:
			self._row_inds = list(
				set(self._row_inds)
				.intersection(all_check[all_check].index.to_list())
			)

		else:
			raise ValueError(f"axis must be 0 or 1, got {axis}")

	OTHER_T = TypeVar("OTHER_T", bound = "Data2D")

	def subset_shared(
		self,
		other: "OTHER_T | Data2DView[OTHER_T]",
		shared_rows: bool = False,
		shared_cols: bool = False,
		self_drop_nan_rows: Optional[HOW_T] = None,
		self_drop_nan_columns: Optional[HOW_T] = None,
		other_drop_nan_rows: Optional[HOW_T] = None,
		other_drop_nan_columns: Optional[HOW_T] = None
	) -> "Tuple[Data2DView[REAL_T], Data2DView[OTHER_T]]":
		"""
		"""
		if not (shared_rows or shared_cols):
			raise ValueError(
				"One or both of shared_rows and shared_cols must be True"
			)

		## Get rows to subset

		if shared_rows:
			self_row_names = list(
				set(self.row_names).intersection(other.row_names)
			)

			if len(self_row_names) == 0:
				raise ValueError(
					f"No shared row names between {self.data_name} and "
					f"{other.data_name} objects."
				)

			other_row_names = self_row_names

		else:
			self_row_names = self.row_names
			other_row_names = other.row_names

		## Get columns to subset

		if shared_cols:
			self_col_names = list(
				set(self.col_names).intersection(other.col_names)
			)

			if len(self_col_names) == 0:
				raise ValueError(
					f"No shared column names between {self.data_name} and "
					f"{other.data_name} objects."
				)

			other_col_names = self_col_names

		else:
			self_col_names = self.col_names
			other_col_names = other.col_names

		self_subs = self.subset(
			row_names = self_row_names,
			column_names = self_col_names,
			drop_nan_rows = self_drop_nan_rows,
			drop_nan_columns = self_drop_nan_columns
		)

		other_subs = other.subset(
			row_names = other_row_names,
			column_names = other_col_names,
			drop_nan_rows = other_drop_nan_rows,
			drop_nan_columns = other_drop_nan_columns
		)

		return (self_subs, other_subs)

	@overload
	def subset_random_samples(
		self,
		frac: float,
		rng: np.random.Generator,
		return_both: Literal[False],
		no_zero_rows: bool = True,
		max_tries: int = 100
	) -> "Data2DView[REAL_T]": ...
	@overload
	def subset_random_samples(
		self,
		frac: float,
		rng: np.random.Generator,
		return_both: Literal[True],
		no_zero_rows: bool = True,
		max_tries: int = 100
	) -> "Tuple[Data2DView[REAL_T], Data2DView[REAL_T]]": ...
	def subset_random_samples(
		self,
		frac: float,
		rng: np.random.Generator,
		return_both: bool = False,
		no_zero_rows: bool = True,
		max_tries: int = 100
	) -> "Data2DView[REAL_T] | Tuple[Data2DView[REAL_T], Data2DView[REAL_T]]":
		if (frac < 0.0) or (frac > 1.0):
			raise ValueError(
				"Fraction of Expression sampled must be between 0 and 1"
			)

		## Define loop stop checks
		## Convention: 'keep' is first matrix, 'disc' (discard) is second matrix

		keep_has_zero_rows = True
		disc_has_zero_rows = True
		tries = 0

		## Define sampling checks
		n_to_choose = int(self.shape[1] * frac)

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
						f"to fraction {frac} of {self.shape[1]} samples."
					))

			else:
				break

		if return_both: 
			return keep, disc
		else:
			return keep

	def squeeze(
		self
	) -> List[float]:
		"""
		"""
		if self.shape[0] == 1:
			return self._data2d.data.iloc[0,self._col_inds].to_list()

		elif self.shape[1] == 1:
			return self._data2d.data.iloc[self._row_inds,0].to_list()

		else:
			raise ValueError(
				f"Cannot call squeeze() for data of shape {self.shape}."
			)

	def has_zero_row(self) -> bool:
		return min(self.data.sum(axis = 1)) == 0

	@property
	def data_name(self) -> str:
		return self._data2d.data_name

	@property
	def row_title(self) -> str:
		return self._data2d.row_title

	@property
	def col_title(self) -> str:
		return self._data2d.col_title

	@property
	def row_names(self) -> List[str]:
		return self._data2d.data.index[self._row_inds].to_list()

	@property
	def col_names(self) -> List[str]:
		return self._data2d.data.columns[self._col_inds].to_list()

	@property
	def data(self) -> pd.DataFrame:
		return self._data2d.data.iloc[self._row_inds, self._col_inds]

	@property
	def shape(self) -> Tuple[int, int]:
		return (len(self._row_inds), len(self._col_inds))

	@property
	def data2d(self) -> REAL_T:
		return self._data2d

	@property
	def array(self) -> np.ndarray:
		return self._data2d._data.iloc[self._row_inds, self._col_inds].to_numpy()




