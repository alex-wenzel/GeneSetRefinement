"""
Module for cophenetic coefficient functions as (usually) performed
on the `CombinedGeneComponentIC` matrix.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.decomposition import NMF
import sys
from typing import cast, List, Optional, Tuple, TYPE_CHECKING

## multiprocess doesn't have type hints, use the ones from 
## multiprocessing as a close approximation during development
if TYPE_CHECKING: 
	import multiprocessing as mp
else:
	import multiprocessing as mp


def _np2df(
	mat: np.ndarray,
	index: Optional[List[str]] = None,
	columns: Optional[List[str]] = None
) -> pd.DataFrame:
	"""
	Converts a matrix into a Pandas DataFrame using the given
	column and index.
	"""
	if index is None:
		index = list(map(str, range(mat.shape[0])))
	
	if columns is None:
		columns = list(map(str, range(mat.shape[1])))

	return pd.DataFrame(
		mat,
		index = index,
		columns = columns
	)

def _one_iteration(
	X: pd.DataFrame,
	k: int,
	seed_iter: int,
	total: int,
	n_cores: int
) -> pd.DataFrame:
	"""
	Returns a binary connectivity matrix for a single iteration
	"""
	current = mp.current_process()

	## Initialize a single connectivity matrix
	conn_mat = pd.DataFrame(
		data = np.zeros((X.shape[1], X.shape[1])),
		index = X.columns,
		columns = X.columns
	)

	## Get NMF solution
	nmf = NMF(
		n_components = k,
		init = "random",
		solver = "mu",
		beta_loss = "kullback-leibler",
		max_iter = 2000,
		random_state = seed_iter
	)

	## Get H
	nmf.fit_transform(X)
	H = _np2df(
		nmf.components_, 
		columns = X.columns.tolist()
	)

	## Get cluster assignments
	cluster_vec = H.idxmax(axis=0)

	for j_sample, j_cluster in cluster_vec.items():
		for k_sample, k_cluster in cluster_vec.items():
			if j_cluster == k_cluster:
				conn_mat.loc[j_sample, k_sample] += 1 #type: ignore

	return conn_mat

def _get_seed(
	rng: np.random.Generator
) -> int:
	"""
	"""
	return cast(
		int,
		rng.integers(
			low = 1,
			high = np.iinfo(np.int32).max
		)
	)

def get_nmf_consensus_cophenetic(
	X: pd.DataFrame,
	k: int,
	rng: np.random.Generator,
	iterations: int = 100,
	n_cores: int = 1
) -> float:
	"""
	Computes the cophenetic coefficient of the NMF consensus clustering results on the 
    input matrix X. 
	"""
	## shift for NMF
	use_X = X + 1.0

	## Instantiate connectivty matrix
	conn_mat = pd.DataFrame(
		data = np.zeros((use_X.shape[1], use_X.shape[1])),
		index = use_X.columns,
		columns = use_X.columns
	)

	## Get NMF results

	if n_cores > 1:
		bin_conn_mats: List[pd.DataFrame] = []
		args: List[Tuple[pd.DataFrame, int, int, int, int]] = []

		for i in range(iterations):
			args.append((
				use_X.copy(deep = True),
				k,
				_get_seed(rng),
				iterations,
				n_cores
			))

		with mp.Pool(n_cores) as p:
			bin_conn_mats += p.starmap(_one_iteration, args)

		#sys.stderr.write('\n')
		#sys.stderr.flush()

		for bin_mat in bin_conn_mats:
			conn_mat += bin_mat
	elif n_cores == 1:
		for i in range(iterations):
			conn_mat += _one_iteration(
				use_X,
				k,
				_get_seed(rng),
				iterations,
				n_cores
			)

	else:
		raise ValueError(
			f"n_cores must be 1 or greater, got {n_cores}"
		)
	
	conn_mat = conn_mat/iterations

	Z = linkage(conn_mat, "ward")

	c, _ = cophenet(Z, pdist(conn_mat))

	return c
	
