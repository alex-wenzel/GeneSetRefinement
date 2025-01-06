from gp.data import GCT
from multiprocessing.pool import Pool
import numpy as np
from numpy import in1d, full, absolute
from numpy.random import random_sample, sample, seed 
import pandas as pd
from scipy.signal import convolve2d
from scipy.sparse import coo_matrix
from scipy.stats import gaussian_kde, pearsonr
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, TypeVar
import warnings

from .Data2D import Data2D

if TYPE_CHECKING:
	from .GeneSet import GeneSet
	from .Expression import Expression

EPS = np.finfo(float).eps


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


def compute_information_coefficient(
	x: List[float],
	y: List[float],
	n_grids: int = 25,
	jitter: float = 1e-10,
	random_seed: float = 20121020,
	fft: bool = True
) -> float:
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

	Returns
	-------
	`float`
		The information coefficient, bounded between `-1.0` and `1.0`. 
	"""
	x_arr, y_arr = _drop_nan_columns([np.asarray(x), np.asarray(y)])

	if (x_arr == y_arr).all():
		return 1.0
	else:
		try:
			# Need at least 3 values to compute bandwidth
			if len(x_arr) < 3 or len(y_arr) < 3:
				return 0.0
		except TypeError:
			# If x and y are numbers, we cannot continue and IC is zero.
			return 0.0

		x_arr = np.asarray(x, dtype = float)
		y_arr = np.asarray(y, dtype = float)

		# Add jitter
		seed(random_seed)
		x_arr += random_sample(x_arr.size) * jitter
		y_arr += random_sample(y_arr.size) * jitter

		# Compute bandwidths
		pearson_res = pearsonr(x, y)
		cor = pearson_res.correlation

		if fft:
			# Compute the PDF
			fxy = _fastkde(
				x_arr.tolist(),
				y_arr.tolist(),
				gridsize = (n_grids, n_grids)
			)[0]

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
			values = np.vstack([x, y])
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


def run_ssgsea_parallel(
	gene_x_sample: Expression,
	gene_sets: List[GeneSet],
	statistic: str = "ks",
	n_job: int = 1
) -> ssGSEAResult:
	"""
	"""
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

	score__gene_set_x_sample = score__gene_set_x_sample[gene_x_sample.sample_names]

	return ssGSEAResult(score__gene_set_x_sample)






















