from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
from typing import (
	Dict, Generic, List, Literal, Optional, overload, Tuple,
	TypeVar
)
from typing_extensions import Self


class Data2DAbs(metaclass=ABCMeta):

	@abstractmethod
	def _check_subset_rows(
		self,
		subset_labels: List[str]
	) -> Optional[List[str]]:
		pass

	@abstractmethod
	def _check_subset_columns(
		self,
		subset_labels: List[str]
	) -> Optional[List[str]]:
		pass

	@abstractmethod
	def _get_row_inds(
		self,
		row_names: List[str]
	) -> List[int]:
		pass

	@abstractmethod
	def _get_col_inds(
		self,
		column_names: List[str]
	) -> List[int]:
		pass


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

		if isinstance(rows, str):
			rows = [rows]

		if isinstance(cols, str):
			cols = [cols]

		shared_rows = self._check_subset_rows(rows)

		if shared_rows is None:
			raise KeyError((
				f"None of requested {self.row_title} names "
				f"are in this {self.data_name}."
			))

		shared_cols = self._check_subset_columns(cols)

		if shared_cols is None:
			raise KeyError((
				f"None of requested {self.col_title} names "
				f"are in this {self.data_name}."
			))

		filt_data: pd.DataFrame = self.data.loc[shared_rows, shared_cols]

		return self.__class__(filt_data)
	
	def _check_subset_rows(
		self,
		subset_labels: List[str]
	) -> Optional[List[str]]:
		"""
		"""
		found_labels: List[str] = []

		for label in subset_labels:
			try:
				self._data.loc[label,:]
				found_labels.append(label)
			except KeyError:
				continue

		if len(found_labels) == 0:
			return None
		
		return found_labels
		
	def _check_subset_columns(
		self,
		subset_labels: List[str]
	) -> Optional[List[str]]:
		"""
		"""
		found_labels: List[str] = []

		for label in subset_labels:
			try:
				self._data.loc[:,label]
				found_labels.append(label)
			except KeyError:
				continue

		if len(found_labels) == 0:
			return None
		
		return found_labels
	
	def _get_row_inds(
		self, 
		row_names: List[str]
	) -> List[int]:
		all_row_names: List[str] = self._data.index.to_list()

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
		all_col_names: List[str] = self._data.columns.to_list()

		col_inds: List[int] = []

		for cn in column_names:
			try:
				col_inds.append(all_col_names.index(cn))
			except ValueError:
				continue

		return col_inds

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
		ret_rows: Optional[List[str]] = row_names
		ret_columns: Optional[List[str]] = column_names

		## Get shared row names

		if len(ret_rows) == 0:
			ret_rows = self.row_names
		else:
			ret_rows = self._check_subset_rows(ret_rows)

			if ret_rows is None:
				raise KeyError((
					f"None of requested {self.row_title} names "
					f"are in this {self.data_name}."
				))
			
		## Get shared column names

		if len(ret_columns) == 0:
			ret_columns = self.col_names
		else:
			ret_columns = self._check_subset_columns(ret_columns)

			if ret_columns is None:
				raise KeyError((
					f"None of requested {self.col_title} names "
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

		keep_has_zero_row = True
		disc_has_zero_row = True
		tries = 0

		## Define sampling checks
		n_to_choose = int(self.shape[1] * frac)

		# Save column order
		col_order: Dict[str, int] = {cn: i for i, cn in enumerate(self.col_names)}

		## Sampling loop
		while True:
			## Choose columns for 'keep' matrix
			keep_cols = rng.choice(
				self.col_names,
				size = n_to_choose,
				replace = False
			).tolist()

			keep_cols.sort(key = lambda cn: col_order[cn])

			## Build keep and discard matrices
			disc_cols = [
				col for col in self.col_names
				if col not in keep_cols
			]

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
		return self._data.index.to_list()

	@property
	def col_names(self) -> List[str]: 
		return self._data.columns.to_list()

	@property
	def data(self) -> pd.DataFrame:
		return self._data

	@property
	def shape(self) -> Tuple[int, int]:
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

		if isinstance(rows, str):
			rows = [rows]

		if isinstance(cols, str):
			cols = [cols]

		shared_rows = self._check_subset_rows(rows)

		if shared_rows is None:
			raise KeyError((
				f"None of requested {self.row_title} names "
				f"are in this {self.data_name}."
			))

		shared_cols = self._check_subset_columns(cols)

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
	
	def _check_subset_rows(
		self,
		subset_labels: List[str]
	) -> Optional[List[str]]:
		"""
		"""
		row_name_d = {
			row_name: True for row_name in self.row_names
		}

		found_labels: List[str] = []

		for label in subset_labels:
			try:
				row_name_d[label]
				found_labels.append(label)
			except KeyError:
				continue

		if len(found_labels) == 0:
			return None
		
		return found_labels
	
	def _check_subset_columns(
		self,
		subset_labels: List[str]
	) -> Optional[List[str]]:
		"""
		"""
		col_name_d = {
			col_name: True for col_name in self.col_names
		}

		found_labels: List[str] = []

		for label in subset_labels:
			try:
				col_name_d[label]
				found_labels.append(label)
			except KeyError:
				continue

		if len(found_labels) == 0:
			return None
		
		return found_labels
	
	def _get_row_inds(
		self,
		row_names: List[str]
	) -> List[int]:
		all_row_names: List[str] = self._data2d._data.index.to_list()

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
		all_col_names: List[str] = self._data2d._data.columns.to_list()

		col_inds: List[int] = []

		for cn in column_names:
			try:
				col_inds.append(all_col_names.index(cn))
			except ValueError:
				continue

		return col_inds

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
		else:
			ret_rows = [
				row_name for row_name in ret_rows
				if row_name in self.row_names
			]

			if len(ret_rows) == 0:
				raise KeyError((
					f"None of requested {self.row_title} names "
					f"are in this {self.data_name}."
				))
			
		## Get shared column names

		if len(ret_columns) == 0:
			ret_columns = self.col_names
		else:
			ret_columns = [
				column_name for column_name in ret_columns
				if column_name in self.col_names
			]

			if len(ret_columns) == 0:
				raise KeyError((
					f"None of requested {self.col_title} names "
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
			col_inds_order = {ind: i for i, ind in enumerate(self._col_inds)}

			self._col_inds = list(
				set(self._col_inds)
				.intersection(all_check[all_check].index.to_list())
			)

			self._col_inds.sort(key = lambda ind: col_inds_order[ind])

		elif axis == 1:
			row_inds_order = {ind: i for i, ind in enumerate(self._row_inds)}

			self._row_inds = list(
				set(self._row_inds)
				.intersection(all_check[all_check].index.to_list())
			)

			self._row_inds.sort(key = lambda ind: row_inds_order[ind])

		else:
			raise ValueError(f"axis must be 0 or 1, got {axis}")

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

		keep_has_zero_row = True
		disc_has_zero_row = True
		tries = 0

		## Define sampling checks
		n_to_choose = int(self.shape[1] * frac)

		## Sampling loop
		while True:
			## Choose columns for 'keep' matrix
			keep_col_inds = rng.choice(
				self._col_inds,
				size = n_to_choose,
				replace = False
			).tolist()

			keep_col_inds = sorted(keep_col_inds)

			## Build keep and discard matrices
			disc_col_inds = [
				col_ind for col_ind in self._col_inds
				if col_ind not in keep_col_inds
			]

			keep = Data2DView(self._data2d, self._row_inds, keep_col_inds)
			disc = Data2DView(self._data2d, self._row_inds, disc_col_inds)

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
		if len(self._row_inds) == 1:
			ind = self._row_inds[0]
			return self._data2d.data.iloc[ind, self._col_inds].to_list()
		
		elif len(self._col_inds) == 1:
			ind = self._col_inds[0]
			return self._data2d.data.iloc[self._row_inds, ind].to_list()
		
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




