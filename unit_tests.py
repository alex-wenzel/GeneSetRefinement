"""
Unit tests. Works with DepMap 23Q2 data parsed into GCT files. Expects the
following files to be present at the directory provided in sys.argv[1]:

- processed_solid_samples/depmap23q2_expression_solid_samples.gct
- processed_solid_samples/depmap23q2_rppa_solid_samples.gct
- processed_solid_samples/depmap23q2_crispr_solid_samples.gct
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import os
import pandas as pd
from pathlib import Path
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
			f"depmap24q4_solid_expression.gct"
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

		cls.expr._data = cls.expr.data.loc[rnd_genes,:]

	def test_shape(self):
		self.assertEqual(self.expr.n_genes, 100)
		self.assertEqual(self.expr.n_samples, 1424)

	def test_subset(self):
		subs_samples = ["ACH-000696", "ACH-000885"]

		subs = gsr.Data2D.subset(
			self.expr,
			column_names = subs_samples
		)

		self.assertEqual(subs.shape[0], self.expr.n_genes)
		self.assertEqual(subs.shape[1], len(subs_samples))

	def test_subset_keep(self):
		keep_frac = 0.3

		keep = self.expr.subset_random_samples(
			keep_frac,
			self.rng,
			return_both = False,
		)

		self.assertIsInstance(keep, gsr.Data2DView)
		self.assertIsInstance(keep.data2d, gsr.Expression)
		self.assertEqual(
			keep.shape[1],
			int(self.expr.shape[1] * keep_frac)
		)

	def test_subset_keep_disc(self):
		keep_frac = 0.3

		subs_res = self.expr.subset_random_samples(
			keep_frac,
			self.rng,
			return_both = True
		)

		self.assertIsInstance(subs_res, tuple)
		keep, disc = subs_res

		self.assertTrue(
			abs(keep.shape[1] - int(self.expr.shape[1] * keep_frac)) <= 1
		)

		self.assertTrue(
			abs(disc.shape[1] - int(self.expr.shape[1] * (1.0 - keep_frac)) <= 1)
		)

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
			f"depmap24q4_solid_protein-rppa.gct"
		)

		paths_d = {
			"rppa": rppa_path,
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

	def test_to_gmt_row(self):
		gs = gsr.GeneSet("a_gene_set", ["a", "b", "c", "d"])

		gmt_row_desc = gs.to_gmt_row(description = "about_a_gene_set")
		gmt_row_no_desc = gs.to_gmt_row()

		self.assertEqual(
			gmt_row_desc,
			"a_gene_set\tabout_a_gene_set\ta\tb\tc\td"
		)

		self.assertEqual(
			gmt_row_no_desc,
			"a_gene_set\ta_gene_set\ta\tb\tc\td"
		)


class RefinementTests(unittest.TestCase):
	expression_path: str
	rppa_path: str
	paths_d: Dict[str, str]
	gene_set_path: str
	gene_set_name: str

	ref: gsr.Refinement

	@classmethod
	def setUpClass(cls):
		depmap_path = sys.argv[1]

		if depmap_path[-1] != '/':
			depmap_path += '/'

		cls.expression_path = (
			f"{depmap_path}/processed_solid_samples/"
			f"depmap24q4_solid_expression.gct"
		)

		cls.rppa_path = (
			f"{depmap_path}/processed_solid_samples/"
			f"depmap24q4_solid_protein-rppa.gct"
		)

		cls.paths_d = {
			"rppa": cls.rppa_path,
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
			n_outer_iterations = 2,
			n_inner_iterations = 2,
			min_total_gene_counts = 5,
			cutoff = 0.3,
			gsea_statistic = "auc",
			random_seed = 49,
			normalize_expression = True,
			max_downsample_tries = 100,
			n_proc = 2,
			verbose = True
		)

		cls.ref.preprocess()

		cls.ref.run()

		cls.one_ii = cls.ref.iterations[3][0][0]

		cls.true_ref = gsr.Refinement.load(
			(
				"examples/example_results/"
				"reactome-erbb2-60_k23_o2_i2_s49_24q4.pickle"
			)
		)

	def test_true_gene_sets(self) -> None:
		for k in self.ref.k_values:
			for i in range(k):
				true_comp_gs = (
					self.true_ref.component_clusters[k][i].gene_set.genes
				)
				ref_comp_gs = self.ref.component_clusters[k][i].gene_set.genes

				self.assertSetEqual(
					set(true_comp_gs),
					set(ref_comp_gs)
				)

	def _data2d_is_close(
		self,
		data1: pd.DataFrame,
		data2: pd.DataFrame
	) -> bool:
		arr1 = data1.to_numpy().flatten()
		arr2 = data2.to_numpy().flatten()

		if len(arr1) != len(arr2):
			raise ValueError((
				f"Cannot compare equality of {data1.data_name} "
				f"with shape {data1.shape} and {data2.data_name} "
				f"with shape {data2.shape}."
			))

		return all([
			np.isclose(arr1[i], arr2[i])
			for i in range(len(arr1))
		])
	
	def test_true_W_matrix(self) -> None:
		true_first_W = self.true_ref._iterations[3][0][0].W.data
		ref_first_W = self.ref._iterations[3][0][0].W.data

		self.assertTrue(self._data2d_is_close(true_first_W, ref_first_W))

	def test_true_gene_comp_IC(self) -> None:
		true_first_gcic = self.true_ref._iterations[3][0][0].gene_component_IC.data
		ref_first_gcic = self.ref._iterations[3][0][0].gene_component_IC.data

		self.assertTrue(self._data2d_is_close(true_first_gcic, ref_first_gcic))

	def test_true_ssGSEAResult(self) -> None:
		true_first_ssgsea = self.true_ref.ssgsea_res[3].data
		ref_first_ssgsea = self.ref.ssgsea_res[3].data

		self.assertTrue(self._data2d_is_close(true_first_ssgsea, ref_first_ssgsea))

	def test_true_phencompic(self) -> None:
		true_first_phenic = self.true_ref.phenotype_component_ics[3][0]["rppa"].data
		ref_first_phenic = self.ref.phenotype_component_ics[3][0]["rppa"].data

		self.assertTrue(self._data2d_is_close(true_first_phenic, ref_first_phenic))

	def test_instantiating_refinement(self):
		self.assertListEqual(self.ref.k_values, [2, 3])

	def test_W_matrix_good_subset(self):
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
			some_known_shared_genes = self.one_ii.W.row_names[:2]

			self.one_ii.W.subset(
				some_known_shared_genes,
				["asfd", "adfsdf"]
			)

	def test_gene_comp_ic_shape(self):
		self.assertEqual(
			self.one_ii.gene_component_IC.shape[0],
			self.one_ii.W.shape[0]
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
			self.one_ii.W.shape[0],
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
				self.ref.expression.shape[1]
			)

	def test_phen_comp_ics(self):
		one_phen_comp = self.ref.phenotype_component_ics[3][0]["rppa"]
		true_one_phen_comp = self.true_ref.phenotype_component_ics[3][0]["rppa"]

		self.assertTrue(
			self._data2d_is_close(
				one_phen_comp.data,
				true_one_phen_comp.data
			)
		)

	def test_io(self):
		out_path = "_test_ref_out.pickle"

		self.assertTrue( 
			self.ref.assert_not_None(self.ref._expr)
		)

		self.ref.save(
			out_path, 
			remove_inputs = True
		)

		load_obj = gsr.Refinement.load(out_path)
		os.remove(out_path)

		self.assertEqual(self.ref._version, load_obj._version)

		with self.assertRaises(gsr.Refinement.RefinementNoneException):
			load_obj.assert_not_None(load_obj._expr)

	def test_phencomp_exception(self):
		## This test looks circular but the point is to make sure all the 
		## parsing in the exception works and doesn't create additional exceptions.
		e = ValueError("an example error")
		
		fake_ssgsea_res = gsr.Data2DView(
			gsr.ssGSEAResult(
				pd.DataFrame({
					"sample_1": [1, 2, 3], 
					"sample_2": [3, 4, 5],
				}, index = ["gene_set_1", "gene_set_2", "gene_set_3"]),
			), [0, 1, 2], [0, 1]
		)
		
		fake_phen_vec = gsr.Data2DView(
			gsr.Phenotype(
				pd.DataFrame({
					"sample_1": [1],
					"sample_2": [2]
				}, index = ["phen_1"]),
				phen_name = "test_phen"
			), [0], [0, 1]
		)

		gene_set_name = "gene_set_1"

		phen_name = "phen_df_1"

		phen_feature_name = "phen_1"

		gene_set = gsr.GeneSet("gene_set_1", ["a", "b", "c", "d"])

		with self.assertRaises(gsr.PhenotypeComponentIC.PhenCompNumericException):
			raise gsr.PhenotypeComponentIC.PhenCompNumericException(
				e,
				fake_ssgsea_res,
				fake_phen_vec,
				gene_set_name,
				phen_name,
				phen_feature_name,
				gene_set
			)
		
	def test_to_gmt(self):
		never_run_ref = gsr.Refinement(
			self.expression_path,
			self.paths_d,
			self.gene_set_path,
			"REACTOME_SIGNALING_BY_ERBB2_v6.0",
			[2, 3],
			n_outer_iterations=2,
			n_inner_iterations=2,
			verbose = True
		)

		with self.assertRaisesRegex(
			RuntimeError,
			".*Has refinement been run?"
		):
			never_run_ref.to_gmt("never_write_me.gmt")

		self.ref.to_gmt("_test_write_out.gmt")
		gsr.read_gmt("_test_write_out.gmt")
		os.remove("_test_write_out.gmt")


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

	def test_compute_information_coefficient_exceptions(self):
		rng = np.random.default_rng(49)

		## bad lengths
		vec1 = [round(rng.random(), ndigits = 2) for _ in range(10)]
		vec2 = [round(rng.random(), ndigits = 2) for _ in range(11)]

		with self.assertRaisesRegex(ValueError, ".*equal length.*"):
			gsr.compute_information_coefficient(vec1, vec2)

		self.assertIsNone(
			gsr.compute_information_coefficient(vec1, vec2, raise_if_failed = False)
		)

		## too short

		vec1 = [round(rng.random(), ndigits = 2) for _ in range(2)]
		vec2 = [round(rng.random(), ndigits = 2) for _ in range(2)]

		with self.assertRaisesRegex(ValueError, ".*must have at least three.*"):
			gsr.compute_information_coefficient(vec1, vec2)

		self.assertIsNone(
			gsr.compute_information_coefficient(vec1, vec2, raise_if_failed = False)
		)

	def test_compute_information_coefficient_nans(self):
		rng = np.random.default_rng(49)

		vec1 = [round(rng.random(), ndigits = 2) for _ in range(10)]
		vec2 = [round(rng.random(), ndigits = 2) for _ in range(10)]

		vec1[6] = np.nan
		vec1[8] = np.nan

		vec2[5] = np.nan
		vec2[7] = np.nan

		self.assertIsInstance(
			gsr.compute_information_coefficient(vec1, vec2),
			float
		)

		for i in range(8):
			vec2[i] = np.nan

		with self.assertRaisesRegex(ValueError, ".*must have at least three.*"):
			gsr.compute_information_coefficient(vec1, vec2)


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

	@classmethod
	def setUpClass(cls):
		cls.test_data: pd.DataFrame = pd.DataFrame(
			{"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
			index = ["r1", "r2", "r3"]
		)

		cls.good_obj = cls.GoodRealData(cls.test_data)
		cls.bad_obj = cls.BadRealData()

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


#class VersionTest(unittest.TestCase):
#	def test_version(self):
#		with open("src/GeneSetRefinement/_version.py", 'r') as f:
#			text_version = f.readline().strip('\n').split('=')[1].strip()[1:-1]
#
#		self.assertEqual(text_version, gsr.__version__)


if __name__ == "__main__":
	## https://stackoverflow.com/questions/2812218/
	## problem-with-sys-argv1-when-unittest-module-is-in-a-script
	unittest.main(
		argv = [sys.argv[0]],
		verbosity = 2,
		defaultTest = [
			"ExpressionTests",
			"PhenotypesTests",
			"GeneSetTests",
			"RefinementTests",
			"UtilsTests",
			"Data2DTests",
			
			##"VersionTest"
		]
	)





