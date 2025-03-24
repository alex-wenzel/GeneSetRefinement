"""
Unit tests. Works with DepMap 23Q2 data parsed into GCT files. Expects the
following files to be present at the directory provided in sys.argv[1]:

- processed_solid_samples/depmap23q2_expression_solid_samples.gct
- processed_solid_samples/depmap23q2_rppa_solid_samples.gct
- processed_solid_samples/depmap23q2_crispr_solid_samples.gct
"""

import numpy as np
import os
import pandas as pd
import string
import sys
from typing import Dict
import unittest

import src.GeneSetRefinement as gsr


class ExpressionTests(unittest.TestCase):
	expr: gsr.Expression
	rng: np.random.Generator

	@classmethod
	def setUpClass(cls):
		depmap_path = sys.argv[1]

		if depmap_path[-1] != '/':
			depmap_path += '/'

		expression_path = (
			f"{depmap_path}/processed_solid_samples/"
			f"depmap23q2_expression_solid_samples.gct"
		)

		cls.expr = gsr.Expression.from_gct(
			expression_path,
			min_counts = -100
		)

		cls.expr.normalize()

		cls.rng = np.random.default_rng(49)

		rnd_genes = cls.rng.choice(
			cls.expr.row_names,
			size = 100,
			replace = False,
		).tolist()

		#cls.expr = gsr.Data2D.subset(
		#	cls.expr, 
		#	row_names = rnd_genes,
		#)

		cls.expr._data = cls.expr.data.loc[rnd_genes,:]

	def test_shape(self):
		self.assertEqual(self.expr.n_genes, 100)
		self.assertEqual(self.expr.n_samples, 1122)

	def test_subset(self):
		subs_samples = ["ACH-000696", "ACH-000885"]

		subs = gsr.Data2D.subset(
			self.expr,
			column_names = subs_samples
		)

		self.assertEqual(subs.shape[0], self.expr.n_genes)
		self.assertEqual(subs.shape[1], len(subs_samples))

	def test_subset_keep(self):
		keep = self.expr.subset_random_samples(
			0.3,
			self.rng,
			return_both = False,
		)

		#self.assertIsInstance(keep, gsr.Expression)
		#self.assertIsInstance(keep, gsr.Data2DView[gsr.Expression])
		self.assertIsInstance(keep, gsr.Data2DView)
		self.assertIsInstance(keep.data2d, gsr.Expression)
		self.assertEqual(keep.shape[1], 336) # type: ignore

	def test_subset_keep_disc(self):
		subs_res = self.expr.subset_random_samples(
			0.3,
			self.rng,
			return_both = True
		)

		self.assertIsInstance(subs_res, tuple)
		keep, disc = subs_res # type: ignore

		self.assertEqual(keep.shape[1], 336)
		self.assertEqual(disc.shape[1], 786)

	def test_normalize(self):
		res = float(self.expr["CDKN2A", "ACH-000696"].array[0,0])

		self.assertEqual(
			round(res, ndigits = 2),
			9270.09
		)

	def test_subset_no_features_found(self):
		with self.assertRaisesRegex(
			KeyError,
			(
				r"None of requested sample names are in "
				r"this gene expression matrix\."
			)
		):
			real_genes = self.expr.row_names[:3]
			bad_columns = ["asdf", "asdfsadf"]
			self.expr[real_genes, bad_columns]


class PhenotypesTests(unittest.TestCase):
	phenotypes: gsr.Phenotypes

	@classmethod
	def setUpClass(cls):
		depmap_path = sys.argv[1]

		if depmap_path[-1] != '/':
			depmap_path += '/'

		rppa_path = (
			f"{depmap_path}/processed_solid_samples/"
			f"depmap23q2_rppa_solid_samples.gct"
		)

		proteomics_path = (
			f"{depmap_path}/processed_solid_samples/"
			f"depmap23q2_proteomics_solid_samples.gct"
		)

		paths_d = {
			"rppa": rppa_path,
			"proteomics": proteomics_path
		}

		cls.phenotypes = gsr.Phenotypes.from_gcts(paths_d)

	def test_select_phen_vec(self):
		vec = self.phenotypes.get_phenotype_vector(
			"rppa",
			"Annexin_VII",
			samples = ["ACH-000698", "ACH-000489", "ACH-000431"]
		)

		vec = [round(val, ndigits=2) for val in vec.values]

		self.assertListEqual(vec, [0.24, 0.48, -0.41])


class GeneSetTests(unittest.TestCase):
	gene_set_path: str

	@classmethod
	def setUpClass(cls):
		cls.gene_set_path = (
			"examples/input_gene_sets/REACTOME_SIGNALING_BY_ERBB2_v6.0.gmt"
		)

	def test_read_gmt(self):
		gmt_d = gsr.read_gmt(self.gene_set_path)
		
		self.assertIn("ADCY4", gmt_d["REACTOME_SIGNALING_BY_ERBB2_v6.0"])

	def test_from_gmt(self):
		gs_d = gsr.GeneSet.from_gmt(self.gene_set_path)

		self.assertIn("ADCY4", gs_d["REACTOME_SIGNALING_BY_ERBB2_v6.0"].genes)


class RefinementTests(unittest.TestCase):
	expression_path: str
	rppa_path: str
	proteomics_path: str
	paths_d: Dict[str, str]
	gene_set_path: str
	gene_set_name: str

	ref: gsr.Refinement

	#def setUp(self):
	@classmethod
	def setUpClass(cls):
		#super(RefinementTests).setUpClass()

		depmap_path = sys.argv[1]

		if depmap_path[-1] != '/':
			depmap_path += '/'

		cls.expression_path = (
			f"{depmap_path}/processed_solid_samples/"
			f"depmap23q2_expression_solid_samples.gct"
		)

		cls.rppa_path = (
			f"{depmap_path}/processed_solid_samples/"
			f"depmap23q2_rppa_solid_samples.gct"
		)

		cls.proteomics_path = (
			f"{depmap_path}/processed_solid_samples/"
			f"depmap23q2_proteomics_solid_samples.gct"
		)

		cls.paths_d = {
			"rppa": cls.rppa_path,
			#"proteomics": cls.proteomics_path
		}

		cls.gene_set_path = (
			"examples/input_gene_sets/REACTOME_SIGNALING_BY_ERBB2_v6.0.gmt"
		)

		cls.gene_set_name = "REACTOME_SIGNALING_BY_ERBB2_v6.0"

		cls.ref = gsr.Refinement(
			cls.expression_path,
			cls.paths_d,
			cls.gene_set_path,
			"REACTOME_SIGNALING_BY_ERBB2_v6.0",
			[2, 3],
			n_outer_iterations=2,
			n_inner_iterations=2,
			verbose = True
		)

		cls.ref.run()

		cls.one_ii = cls.ref.iterations[3][0][0]

	def test_instantiating_refinement(self):
		self.assertListEqual(self.ref.k_values, [2, 3])

	def test_inner_iteration_instantiation(self):
		self.assertEqual(
			self.one_ii.generating_expression.shape[1],
			int(self.one_ii.training_expression.shape[1] * 0.50)
		)

	def test_inner_iteration_get_A(self):
		self.assertEqual(
			self.one_ii.A.shape[0],
			95
		)

	def test_A_matrix_bad_subset(self):
		with self.assertRaisesRegex(
			KeyError,
			(
				r"None of requested sample names are in "
				r"this A matrix\."
			)
		): 
			#gsr.Data2D.subset(
			#	self.one_ii.A,
			#	["CDKN2A", "ERBB2"],
			#	["asdf", "asfasf"]
			#)
			self.one_ii.A.subset(
				["CDKN2A", "ERBB2"],
				["asdf", "asfasf"]
			)

	def test_good_A_matrix_subset(self):
		#subs_a = gsr.Data2D.subset(
		#	self.one_ii.A,
		#	["CDKN2A", "ERBB2"],
		#	[]
		#)
		subs_a = self.one_ii.A.subset(
			["CDKN2A", "ERBB2"],
			[]
		)

		#self.assertIsInstance(subs_a, gsr.A_Matrix)
		#self.assertIsInstance(subs_a, gsr.Data2DView[gsr.Expression])
		self.assertIsInstance(subs_a, gsr.Data2DView)
		self.assertIsInstance(subs_a.data2d, gsr.Expression)

	def test_inner_iteration_nmf(self):
		self.assertListEqual(
			list(self.one_ii.W.shape),
			[self.one_ii.A.shape[0], self.one_ii.k]
		)

		self.assertListEqual(
			list(self.one_ii.H.shape),
			[self.one_ii.k, self.one_ii.A.shape[1]]
		)

	def test_W_matrix_good_subset(self):
		#subs_w = gsr.Data2D.subset(
		#	self.one_ii.W,
		#	["CDKN1A", "ERBB2"],
		#	["0", "2"]
		#)
		subs_w = self.one_ii.W.subset(
			["CDKN1A", "ERBB2"],
			["0", "2"]
		)

		self.assertTupleEqual(
			subs_w.shape,
			(2, 2)
		)

	def test_W_matrix_bad_subset(self):
		with self.assertRaisesRegex(
			KeyError,
			(
				r"None of requested component names are in "
				r"this W matrix\."
			)
		):
			#gsr.Data2D.subset(
			#	self.one_ii.W,
			#	["CDKN2A", "ERBB2"],
			#	["asfd", "adfsdf"]
			#)
			self.one_ii.W.subset(
				["CDKN2A", "ERBB2"],
				["asfd", "adfsdf"]
			)


	def test_gene_comp_ic_shape(self):
		self.assertEqual(
			self.one_ii.gene_component_IC.shape[0],
			self.one_ii.A.shape[0]
		)

		self.assertEqual(
			self.one_ii.gene_component_IC.shape[1],
			self.one_ii.H.shape[0]
		)

	def test_combined_gene_comp_ic_shape(self):
		k = 3

		comb = gsr.CombinedGeneComponentIC.from_refinement(
			k,
			self.ref
		)

		self.assertEqual(
			self.one_ii.A.shape[0],
			comb.shape[0]
		)

		self.assertEqual(
			k * self.ref._n_inner * self.ref._n_outer,
			comb.shape[1]
		)

	def test_component_cluster_separation(self):
		k = 3

		n_comps = sum([
			len(self.ref.component_clusters[k][i].component_names)
			for i in range(k)
		])

		self.assertEqual(
			n_comps,
			k * self.ref._n_inner * self.ref._n_outer
		)

	def test_all_ssgsea_res(self):
		for k in self.ref.k_values:
			ssgsea_res = self.ref.ssgsea_res[k]

			self.assertEqual(
				ssgsea_res.shape[0],
				k + 1
			)

			self.assertEqual(
				ssgsea_res.shape[1],
				self.ref._expr.shape[1]
			)

	def test_phen_comp_ics(self):
		phen_name = self.ref.phenotypes.phenotype_table_names[0]

		for k in self.ref.k_values:
			one_phen_comp = self.ref.phenotype_component_ics[k][0][phen_name]

			self.assertEqual(
				one_phen_comp.shape[0],
				k + 1,
			)

			self.assertEqual(
				one_phen_comp.shape[1],
				self.ref.phenotypes[phen_name].shape[0]
			)

	def test_io(self):
		out_path = "_test_ref_out.pickle"

		self.ref.save(out_path)

		load_obj = gsr.Refinement.load(out_path)

		self.assertEqual(gsr.VERSION, load_obj._version)

		os.remove(out_path)


class UtilsTests(unittest.TestCase):
	def test_compute_information_coefficient(self):
		rng = np.random.default_rng(49)

		vec1 = [round(rng.random(), ndigits = 2) for _ in range(10)]
		vec2 = [round(rng.random(), ndigits = 2) for _ in range(10)]

		v1_v2_res = round(
			gsr.compute_information_coefficient(
				vec1, 
				vec2
			), 
			ndigits = 4
		)

		v1_v1_res = round(
			gsr.compute_information_coefficient(
				vec1, 
				vec1
			), 
			ndigits = 4
		)

		self.assertEqual(v1_v2_res, -0.2046)
		self.assertEqual(v1_v1_res, 1.0)


class Data2DTests(unittest.TestCase):
	## Good implementation of Data2D
	class GoodRealData(gsr.Data2D):
		def __init__(self, data: pd.DataFrame):
			super().__init__(data)
		@property
		def data_name(self) -> str: return "Good Data"
		@property
		def row_title(self) -> str: return "rows"
		@property
		def col_title(self) -> str: return "columns"


	## Bad implementation of Data2D that doesn't call super().__init__()
	class BadRealData(gsr.Data2D):
		def __init__(self) -> None:
			pass
		@property
		def data_name(self) -> str: return "Bad Data"
		@property
		def row_title(self) -> str: return "rows"
		@property
		def col_title(self) -> str: return "columns"

	class DataNewAttr(gsr.Data2D):
		def __init__(self, data: pd.DataFrame, param: int = 5):
			super().__init__(data)
			self._param = param
		@property
		def data_name(self) -> str: return "New Attr Data"
		@property
		def row_title(self) -> str: return "rows"
		@property
		def col_title(self) -> str: return "columns"

	@classmethod
	def setUpClass(cls):
		cls.test_data: pd.DataFrame = pd.DataFrame(
			{"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
			index = ["r1", "r2", "r3"]
		)

		cls.good_obj = cls.GoodRealData(cls.test_data)
		cls.bad_obj = cls.BadRealData()
		cls.new_attr_obj = cls.DataNewAttr(cls.test_data, param = 5)

	def test_getitem(self):
		new_obj = self.good_obj[["r1", "r3", "r4"], ["a", "c"]]

		self.assertListEqual(
			new_obj.row_names,
			["r1", "r3"]
		)

		self.assertListEqual(
			new_obj.col_names,
			["a", "c"]
		)

	def test_subset_nan(self):
		obj = self.GoodRealData(
			pd.DataFrame({
				"a": [0, 		3, 		4,			4],
				"b": [1, 		3, 		np.nan,		3],
				"c": [np.nan,	np.nan, np.nan,		np.nan],
				"d": [2,		3,		np.nan,		1]
			})
		)

		## Test 1
		subs = obj.subset(
			drop_nan_columns="any"
		)
		self.assertTupleEqual(
			subs.shape,
			(4, 1)
		)

		## Test 2
		subs = obj.subset(
			drop_nan_rows = "all"
		)
		self.assertTupleEqual(
			subs.shape,
			(4, 4)
		)

		## Test 3
		subs = obj.subset(
			drop_nan_columns="all"
		)
		self.assertTupleEqual(
			subs.shape,
			(4, 3)
		)

	def test_shared_subset(self):
		obj1 = self.GoodRealData(
			pd.DataFrame({
				"a": [0, 1, 2],
				"b": [3, 4, 5],
				"c": [4, 5, 2],
				"d": [4, 5, 1]
			},
			index = [f"r{i}" for i in range(3)] # r0, r1, r2,
			)
		)

		obj2 = self.GoodRealData(
			pd.DataFrame({
				"b": [4, 5, 1],
				"c": [3, 4, 1],
				"d": [2, 3, 4],
				"e": [1, 2, 4]
			},
			index = [f"r{i}" for i in range(2, 5)] # r2, r3, r4
			)
		)

		## Rows only

		subs_obj1, subs_obj2 = obj1.subset_shared(
			obj2,
			shared_rows = True
		)
		self.assertTupleEqual(
			(subs_obj1.shape, subs_obj2.shape),
			((1,4), (1,4))
		)

		## Columns only

		subs_obj1, subs_obj2 = obj1.subset_shared(
			obj2,
			shared_cols = True
		)
		self.assertTupleEqual(
			(subs_obj1.shape, subs_obj2.shape),
			((3, 3), (3, 3))
		)

		## Rows and columns

		subs_obj1, subs_obj2 = obj1.subset_shared(
			obj2,
			shared_rows = True,
			shared_cols = True
		)
		self.assertTupleEqual(
			(subs_obj1.shape, subs_obj2.shape),
			((1, 3), (1, 3))
		)

	def test_get_row_inds_data2d(self):
		obj = self.GoodRealData(
			pd.DataFrame({
				"a": [0, 1, 2],
				"b": [3, 4, 5],
				"c": [4, 5, 2],
				"d": [4, 5, 1]
			},
			index = [f"r{i}" for i in range(3)] # r0, r1, r2,
			)
		)

		inds = obj._get_row_inds(["r1", "r2"])

		self.assertListEqual(inds, [1, 2])

	def test_get_col_inds_data2d(self):
		obj = self.GoodRealData(
			pd.DataFrame({
				"a": [0, 1, 2],
				"b": [3, 4, 5],
				"c": [4, 5, 2],
				"d": [4, 5, 1]
			},
			index = [f"r{i}" for i in range(3)] # r0, r1, r2,
			)
		)

		inds = obj._get_col_inds(["b", "d"])

		self.assertListEqual(inds, [1, 3])
	
	def test_get_row_inds_data2dview(self):
		obj = self.GoodRealData(
			pd.DataFrame({
				"a": [0, 1, 2],
				"b": [3, 4, 5],
				"c": [4, 5, 2],
				"d": [4, 5, 1]
			},
			index = [f"r{i}" for i in range(3)] # r0, r1, r2,
			)
		)

		subs_obj = obj.subset(row_names = ["r1", "r2"])

		inds = subs_obj._get_row_inds(["r1", "r2"])

		self.assertListEqual(inds, [1, 2])

	def test_get_col_inds_data2dview(self):
		obj = self.GoodRealData(
			pd.DataFrame({
				"a": [0, 1, 2],
				"b": [3, 4, 5],
				"c": [4, 5, 2],
				"d": [4, 5, 1]
			},
			index = [f"r{i}" for i in range(3)] # r0, r1, r2,
			)
		)

		subs_obj = obj.subset(column_names = ["b", "d"])

		inds = subs_obj._get_col_inds(["b", "d"])

		self.assertListEqual(inds, [1, 3])

	def test_row_subset_order(self):
		full_names = [char for char in string.ascii_letters]

		subset_names = [char for char in string.ascii_lowercase[::-1]]

		obj = self.GoodRealData(
			pd.DataFrame({
				char: [1, 2, 3]
				for char in full_names
			}).T
		)

		subs_obj = obj.subset(row_names = subset_names)

		joined_subs_row_names = ''.join(subs_obj.row_names)

		self.assertEqual(joined_subs_row_names, string.ascii_lowercase[::-1])

	def test_col_subset_order(self):
		full_names = [char for char in string.ascii_letters]

		subset_names = [char for char in string.ascii_lowercase[::-1]]

		obj = self.GoodRealData(
			pd.DataFrame({
				char: [1, 2, 3]
				for char in full_names
			})
		)

		subs_obj = obj.subset(column_names = subset_names)

		joined_subs_col_names = ''.join(subs_obj.col_names)

		self.assertEqual(joined_subs_col_names, string.ascii_lowercase[::-1])


if __name__ == "__main__":
	## https://stackoverflow.com/questions/2812218/
	## problem-with-sys-argv1-when-unittest-module-is-in-a-script
	unittest.main(
		argv = [sys.argv[0]],
		verbosity = 2,
		defaultTest = [
		#	"ExpressionTests",
		#	"PhenotypesTests",
		#	"GeneSetTests",
		#	"UtilsTests",
			"Data2DTests"
		]
	)





