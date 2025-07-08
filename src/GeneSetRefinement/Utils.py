from __future__ import annotations

from datetime import datetime
from gp.data import GCT
import json
from multiprocessing import Process, Lock
from multiprocessing.pool import Pool
import numpy as np
from numpy import in1d, full, absolute
from numpy.random import random_sample, seed 
import pandas as pd
import psutil
from scipy.signal import convolve2d
from scipy.sparse import coo_matrix
from scipy.stats import gaussian_kde, pearsonr
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Optional, overload, Tuple, TypeVar
import warnings

from .Data2D import Data2D
from .GeneSet import GeneSet

if TYPE_CHECKING:
	from .Expression import Expression

EPS = np.finfo(float).eps


class Log:
	_verbose: bool
	_ts_format: str
	_base_tabs: int

	def __init__(
		self,
		verbose: bool,
		timestamp_format: str = "[%b %d, %Y %H:%M:%S]",
		base_tabs = 0
	) -> None:
		self._verbose = verbose
		self._ts_format = timestamp_format
		self._base_tabs = base_tabs

	def __call__(
		self,
		msg: str,
		tabs: int = 0,
		always_print: bool = False
	) -> None:
		"""
		"""
		if not (always_print or self._verbose):
			return
		
		log_s = ""
		now = datetime.now()

		log_s += f"{now.strftime(self._ts_format)}"
		log_s += f"{' | ' * (self._base_tabs + tabs)} "
		log_s += f"{msg}"
		
		print(log_s, flush=True)

	@classmethod
	def new_indented_log(
		cls,
		old_log: Log,
		base_tabs: int
	) -> Log:
		"""
		"""
		return cls(
			old_log._verbose,
			old_log._ts_format,
			base_tabs
		)
	

class MemoryLog(Process):
	def __init__(
		self,
		outpath: str,
		lock: Lock, #type: ignore
		freq: int
	) -> None:
		"""
		"""
		Process.__init__(self)
		self._outpath = outpath
		self._lock = lock
		self._freq = freq
	
	def run(self):
		mem_logs_l = []

		while self._lock.acquire(block=False):
			mem = psutil.virtual_memory().used/1e9
			ts = int(datetime.now().timestamp())

			mem_logs_l.append({"ts": ts, "gb": mem})

			with open(self._outpath, 'w') as f:
				json.dump(mem_logs_l, f)

			self._lock.release()
			time.sleep(self._freq)


def load_gct(
	path: str
) -> pd.DataFrame:
	"""
	Loads a GCT file using the GenePattern Python API and drops the 
	Description column. 

	Parameters
	----------
	`path` : `str`
		Filepath to a GCT file. 

	Returns
	-------
	`pd.DataFrame`
		Dataframe for the GCT with only the first index column. 
	"""
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		data = GCT(path)

	data.index = data.index.droplevel("Description")

	return data


"""
Information coefficient + supporting fuctions
https://github.com/alex-wenzel/ccal-noir/blob/master/ccalnoir/information.py
"""

def _drop_nan_columns(
	arrays: List[np.ndarray],
) -> List[np.ndarray]:
	"""
	For each 1-d array provided, filter the entries (columns) that are NaN. 

	Parameters
	----------
	`arrays` : `list` of Numpy array

	Returns
	-------
	`list` of Numpy array
		Each of the arrays in `arrays` with NaN values removed. 
	"""
	if len(arrays[0]) != len(arrays[1]):
		raise ValueError(
			f"Got arrays of unequal lengths {len(arrays[0])} and {len(arrays[1])}."
		)

	try:
		not_nan_filter = np.ones(len(arrays[0]), dtype = bool)

		for a in arrays:
			not_nan_filter &= ~np.isnan(a)

		return [a[not_nan_filter] for a in arrays]

	except TypeError: 
		## assume this function was provided a number instead of a list. 
		return arrays

def _fastkde(
	x: List[float],
	y: List[float],
	gridsize: Optional[Tuple[int, int]] = (200, 200),
	extents: Optional[Tuple[float, float, float, float]] = None,
	nocorrelation: bool = False,
	weights: Optional[np.ndarray] = None,
	adjust: float = 1.0
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
	"""
	The following documentation is adopted from CCALnoir as it was originally
	adopted from https://github.com/mfouesneau/faststats

	A fft-based Gaussian kernel density estimate (KDE)
	for computing the KDE on a regular grid

	Note that this is a different use case than scipy's original
	scipy.stats.kde.gaussian_kde

	IMPLEMENTATION
	--------------

	Performs a gaussian kernel density estimate over a regular grid using a
	convolution of the gaussian kernel with a 2D histogram of the data.

	It computes the sparse bi-dimensional histogram of two data samples where
	*x*, and *y* are 1-D sequences of the same length. If *weights* is None
	(default), this is a histogram of the number of occurences of the
	observations at (x[i], y[i]).
	histogram of the data is a faster implementation than numpy.histogram as it
	avoids intermediate copies and excessive memory usage!


	This function is typically *several orders of magnitude faster* than
	scipy.stats.kde.gaussian_kde.  For large (>1e7) numbers of points, it
	produces an essentially identical result.

	Boundary conditions on the data is corrected by using a symmetric /
	reflection condition. Hence the limits of the dataset does not affect the
	pdf estimate.

	Parameters
	----------
	`x` : `list` of `float`
		The x coordinates of the input data. 

	`y` : `list` of `float`
		The y coordinates of the input data. 

	`gridsize` : `tuple` of `int`, default `(200, 200)`
		Size of the grid. 

	`extents` : Optional `tuple` of four `float`s, default `None`
		Optionally define the extents of the grid with four tuples representing
		`xmin`, `xmax`, `ymin`, `ymax`. 

	`nocorrelation` : `bool`, default `False`
		If True, the correlation between the x and y coords will be ignored
		when preforming the KDE. (default: False)

	`weights` : optional `np.ndarray`, default `None`
		An array of the same shape as `x` & `y` that weights each sample (`x_i`,
		`y_i`) by each value in weights (`w_i`).  Defaults to an array of ones
		the same size as `x` & `y`. (default: `None`)

	`adjust` : `float`
		An adjustment factor for the bandwidth. Bandwidth becomes `bw * adjust`.
	
	Returns
	-------
	`g` : `ndarray`
		A gridded 2D kernel density estimate of the input points.

	`e` : `tuple` of four `float`s
		Extents of `g` as `(xmin, xmax, ymin, ymax)`.
	"""
	## Variable check
	x_arr, y_arr = np.asarray(x), np.asarray(y)

	x_arr, y_arr = np.squeeze(x), np.squeeze(y)

	if x_arr.size != y_arr.size:
		raise ValueError('Input x & y arrays must be the same size!')

	n = x_arr.size

	if weights is None:
		## Default: Weight all points equally
		weights = np.ones(n)
	else:
		weights = np.squeeze(np.asarray(weights))

		if weights.size != x_arr.size:
			raise ValueError((
				'Input weights must be an array of the same '
				'size as input x & y arrays!'
			))

	# Optimize gridsize ------------------------------------------------------
	# Make grid and discretize the data and round it to the next power of 2
	# to optimize with the fft usage
	gridsize_arr = gridsize
	
	if gridsize_arr is None:
		gridsize_arr = np.asarray(
			[np.max((len(x), 512.)), np.max((len(y), 512.))]
		)

	gridsize_arr = 2 ** np.ceil(np.log2(gridsize_arr))

	## Check grid dimensions are integers to address numpy data type errors
	nx: int = gridsize_arr[0]
	ny: int = gridsize_arr[1]

	# Make the sparse 2d-histogram -------------------------------------------
	# Default extents are the extent of the data

	if extents is None:
		xmin, xmax = x_arr.min(), x_arr.max()
		ymin, ymax = y_arr.min(), y_arr.max()
	else:
		xmin, xmax, ymin, ymax = map(float, extents)

	dx = (xmax - xmin) / (nx - 1)
	dy = (ymax - ymin) / (ny - 1)

	# Basically, this is just doing what np.digitize does with one less copy
	# xyi contains the bins of each point as a 2d array [(xi,yi)]

	xyi = np.vstack((x_arr, y_arr)).T
	xyi -= [xmin, ymin]
	xyi /= [dx, dy]
	xyi = np.floor(xyi, xyi).T

	# Next, make a 2D histogram of x & y.
	# Exploit a sparse coo_matrix avoiding np.histogram2d due to excessive
	# memory usage with many points

	grid = coo_matrix(
		(weights, xyi),
		shape = (int(nx), int(ny))
	).toarray()

	# Kernel Preliminary Calculations ---------------------------------------
	# Calculate the covariance matrix (in pixel coords)
	cov = np.cov(xyi)

	if nocorrelation:
		cov[1,0] = 0
		cov[0,1] = 0

	# Scaling factor for bandwidth
	scotts_factor = n ** (-1.0 / 6.0) * adjust # For 2D

	# Make the gaussian kernel ---------------------------------------------

	# First, determine the bandwidth using Scott's rule
	# (note that Silvermann's rule gives the # same value for 2d datasets)
	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore",
			category=RuntimeWarning
		)
		std_devs = np.diag(np.sqrt(cov))
		
	kern_nx, kern_ny = np.round(scotts_factor * 2 * np.pi * std_devs)
	kern_nx = int(kern_nx)
	kern_ny = int(kern_ny)

	# Determine the bandwidth to use for the gaussian kernel
	inv_cov = np.linalg.inv(cov * scotts_factor ** 2)

	# x & y (pixel) coords of the kernel grid, with <x,y> = <0,0> in center
	xx = np.arange(kern_nx, dtype = float) - kern_nx / 2.0
	yy = np.arange(kern_ny, dtype = float) - kern_ny / 2.0
	xx, yy = np.meshgrid(xx, yy)

	# Then evaluate the gaussian function on the kernel grid
	kernel = np.vstack((xx.flatten(), yy.flatten()))
	kernel = np.dot(inv_cov, kernel) * kernel
	kernel = np.sum(kernel, axis = 0) / 2.0
	kernel = np.exp(-kernel)
	kernel = kernel.reshape((kern_ny, kern_nx))

	#---- Produce the kernel density estimate --------------------------------

	# Convolve the histogram with the gaussian kernel
	# use boundary=symm to correct for data boundaries in the kde
	grid = convolve2d(
		grid,
		kernel,
		mode = "same",
		boundary = "symm"
	)

	# Normalization factor to divide result by so that units are in the same
	# units as scipy.stats.kde.gaussian_kde's output.
	norm_factor = 2 * np.pi * cov * scotts_factor ** 2
	norm_factor = np.linalg.det(norm_factor)
	norm_factor = n * dx * dy * np.sqrt(norm_factor)

	# Normalize the result
	grid /= norm_factor

	return grid, (xmin, xmax, ymin, ymax)


def _raise_if_failed(
	raise_if_failed: bool,
	exc: Exception
) -> None:
	"""
	Uses the `raise_if_failed` value from `compute_information_coefficient()`
	to determine if returning None or raising an exception.
	"""
	if raise_if_failed:
		raise exc
	
	return None


@overload
def compute_information_coefficient(
	x: List[float],
	y: List[float],
	n_grids: int = 25,
	jitter: float = 1e-10,
	random_seed: float = 20121020,
	fft: bool = True,
	raise_if_failed: Literal[True] = True
) -> float:
	...
@overload
def compute_information_coefficient(
	x: List[float],
	y: List[float],
	n_grids: int = 25,
	jitter: float = 1e-10,
	random_seed: float = 20121020,
	fft: bool = True,
	*, ## https://stackoverflow.com/questions/59359943/how-to-write-typing-overload-decorator-for-bool-arguments-by-value
	raise_if_failed: Literal[False]
) -> Optional[float]:
	...
@overload ## https://stackoverflow.com/questions/59359943/how-to-write-typing-overload-decorator-for-bool-arguments-by-value
def compute_information_coefficient(
	x: List[float],
	y: List[float],
	n_grids: int,
	jitter: float,
	random_seed: float,
	fft: bool,
	raise_if_failed: Literal[False]
) -> Optional[float]:
	...
def compute_information_coefficient(
	x: List[float],
	y: List[float],
	n_grids: int = 25,
	jitter: float = 1e-10,
	random_seed: float = 20121020,
	fft: bool = True,
	raise_if_failed: bool = True
) -> Optional[float]:
	"""
	Implementation of the information coefficient based on the CCAL(noir)
	implementation by Kwat Medetgul-Ernar and Edwin Juarez. See
	https://github.com/alex-wenzel/ccal-noir/blob/master/ccalnoir/information.py

	Parameters
	----------
	`x` : `list` of `float`
		The vector to compute the IC with `y`. 

	`y` : `list` of `float`
		The vector to compute the IC with `x`. 

	`n_grids` : `int`, default `25`
		Bandwidth granularity for density estimations. 

	`jitter` : `float`, default `1e-10`
		Small random noise for specific vectors (e.g. highly similar, 
		identities). 

	`random_seed` : `int`, default `20121020`
		Random seed. Will not be changed from the calling Gene Set Refinement
		functions. 

	`fft` : `bool`, default `True`
		If `True`, use `fastkde()` as adopted from 
		https://github.com/mfouesneau/faststats/blob/master/faststats/fastkde.py 
		otherwise scipy's `gaussian_kde()`. 

	`raise_if_failed` : default `True`
		If invalid values are encountered, the appropriate exception will be
		raised if `True`, otherwise the function will silently return `None`.

	Returns
	-------
	`float`
		The information coefficient, bounded between `-1.0` and `1.0`. 
	"""
	x_arr = np.asarray(x, dtype = float)
	y_arr = np.asarray(y, dtype = float)

	try:
		x_arr, y_arr = _drop_nan_columns([x_arr, y_arr])
	except ValueError as e:
		return _raise_if_failed(raise_if_failed, e)
	
	## This should always be true because of how _drop_nan_columns works, but checking just in case. 
	## Unhandled assertion because this being false means there are issues in _drop_nan_columns or elsewhere
	## in case of any future refactoring. 
	assert len(x_arr) == len(y_arr)
	
	if len(x_arr) < 3:
		return _raise_if_failed(
			raise_if_failed,
			ValueError(f"x and y must have at least three elements, got {len(x)} and {len(y)}.")
		)

	if (x_arr == y_arr).all():
		return 1.0

	# Add jitter
	seed(random_seed)
	x_arr += random_sample(x_arr.size) * jitter
	y_arr += random_sample(y_arr.size) * jitter

	# Compute bandwidths
	try:
		pearson_res = pearsonr(x_arr, y_arr)
	except ValueError as e:
		return _raise_if_failed(raise_if_failed, e)
		
	cor = pearson_res.correlation

	if fft:
		# Compute the PDF
		fxy = _fastkde(
			x_arr.tolist(),
			y_arr.tolist(),
			gridsize = (n_grids, n_grids)
		)[0] + EPS

		if fxy.shape[1] != n_grids:
			n_grids = fxy.shape[1]

	else:
		# Estimate fxy using scipy.stats.gaussian_kde()
		xmin = float(x_arr.min())
		xmax = float(x_arr.max())
		ymin = float(y_arr.min())
		ymax = float(y_arr.max())

		X, Y = np.mgrid[
			xmin:xmax:complex(0, n_grids), 
			ymin:ymax:complex(0, n_grids)
		]

		positions = np.vstack([X.ravel(), Y.ravel()])
		values = np.vstack([x_arr, y_arr])
		kernel = gaussian_kde(values)
		fxy = np.reshape(kernel(positions).T, X.shape) + EPS

	dx = (x_arr.max() - x_arr.min()) / (n_grids - 1)
	dy = (y_arr.max() - y_arr.min()) / (n_grids - 1)
	pxy = fxy / (fxy.sum() * dx * dy)
	px = pxy.sum(axis = 1) * dy
	py = pxy.sum(axis = 0) * dx

	# Compute mutual information
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")

		mi = (
			pxy * np.log(
				pxy / ((
					np.asarray([px] * n_grids).T
					* np.asarray([py] * n_grids)
				))
			)
		).sum() * dx * dy 

	# Compute information coefficient
	ic = np.sign(cor) * np.sqrt(1 - np.exp(-2 * mi))

	if np.isnan(ic):
		ic = 0.0

	return ic


## ssGSEA
# https://github.com/alex-wenzel/PheNMF/blob/master/PheNMF/ssGSEAParallel.py

def split_df(
	df: pd.DataFrame,
	axis: int,
	n_split: int
) -> List[pd.DataFrame]:
	"""
	"""
	if not (0 < n_split <= df.shape[axis]):
		raise ValueError(
			f"Invalid: 0 < n_split ({n_split}) <= n_slices ({df.shape[axis]})"
		)

	n = df.shape[axis] // n_split

	dfs = []

	for i in range(n_split):
		start_i = i * n
		end_i = (i + 1) * n

		if axis == 0:
			dfs.append(df.iloc[start_i:end_i])

		elif axis == 1:
			dfs.append(df.iloc[:, start_i:end_i])

	i = n * n_split

	if i < df.shape[axis]:
		if axis == 0:
			dfs.append(df.iloc[i:])

		elif axis == 1:
			dfs.append(df.iloc[:, i:])

	return dfs


RET_T = TypeVar("RET_T")

def multiprocess(
	callable_: Callable[..., RET_T],
	args: List[Any],
	n_job: int,
	random_seed: int = 20121020
) -> List[RET_T]:
	"""
	"""
	seed(random_seed)

	with Pool(n_job) as process:
		return process.starmap(callable_, args)


## Single sample ssGSEA adopted from CCAL and CCALnoir
## Credit: Pablo Tamayo, Kwat Medetgul-Ernar, Edwin Juarez

def _single_sample_gseas(
	gene_x_sample: pd.DataFrame,
	gene_sets: List[GeneSet],
	statistic: str
) -> pd.DataFrame:
	"""
	"""
	score__gene_set_x_sample = full(
		(len(gene_sets), gene_x_sample.shape[1]), 
		np.nan
	)

	for sample_index, (sample_name, gene_score) in enumerate(gene_x_sample.items()):

		for gene_set_index, gene_set in enumerate(gene_sets):
			
			score__gene_set_x_sample[gene_set_index, sample_index] = single_sample_gsea(
				gene_score,
				gene_set.genes,
				statistic = statistic
			)

	score__gene_set_x_sample = pd.DataFrame(
		score__gene_set_x_sample,
		index = [gs.name for gs in gene_sets],
		columns = gene_x_sample.columns
	)

	return score__gene_set_x_sample


def single_sample_gsea(
	gene_score: pd.Series,
	gene_set_genes: List[str],
	statistic: str = "ks",
) -> float: 
	"""
	"""
	gene_score = gene_score.dropna()

	gene_score_sorted = gene_score.sort_values(ascending=False)

	in_ = in1d(
		gene_score_sorted.index,
		gene_set_genes,
		assume_unique=True
	)

	in_sum = int(in_.sum())

	if in_sum == 0:
		return np.nan

	gene_score_sorted_values = list(gene_score_sorted.values)

	gene_score_sorted_values_absolute = absolute(gene_score_sorted_values)

	in_int = in_.astype(int)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		hit = (
			gene_score_sorted_values_absolute * in_int
		) / gene_score_sorted_values_absolute[in_].sum()

		miss = (1 - in_int) / (in_.size - in_sum)

		y = hit - miss

		cumulative_sums = y.cumsum()

	if statistic not in ("ks", "auc"):
		raise ValueError(
			f"Unknown statistic: {statistic}"
		)

	if statistic == "ks":
		max_ = float(cumulative_sums.max())
		min_ = float(cumulative_sums.min())

		if absolute(min_) < absolute(max_):
			score = max_

		else:
			score = min_

	elif statistic == "auc":
		score = cumulative_sums.sum()

	return score


class ssGSEAResult(Data2D):
	@property
	def data_name(self) -> str: return "ssGSEA matrix"
	@property
	def row_title(self) -> str: return "gene set"
	@property
	def col_title(self) -> str: return "sample"


## TODO: Re-parallelize
def run_ssgsea_parallel(
	gene_x_sample: Expression,
	gene_sets: List[GeneSet],
	statistic: str = "auc",
	n_job: int = 1
) -> ssGSEAResult:
	"""
	"""

	#"""
	score__gene_set_x_sample = pd.concat(
		multiprocess(
			_single_sample_gseas,
			[
				(gene_x_sample_, gene_sets, statistic)
				for gene_x_sample_ in split_df(
					gene_x_sample.data, 1, min(gene_x_sample.shape[1], n_job)
				)
			],
			n_job,
		),
		sort = False,
		axis = 1
	)

	score__gene_set_x_sample = score__gene_set_x_sample[gene_x_sample.col_names]

	return ssGSEAResult(score__gene_set_x_sample)
	#"""

	#score__gene_set_x_sample = _single_sample_gseas(
	#	gene_x_sample.data,
	#	gene_sets,
	#	statistic
	#)

	#return ssGSEAResult(score__gene_set_x_sample)






















