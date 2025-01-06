"""
Unit tests. Works with DepMap 23Q2 data parsed into GCT files. Expects the
following files to be present at the directory provided in sys.argv[1]:

- processed_solid_samples/depmap23q2_expression_solid_samples.gct
- processed_solid_samples/depmap23q2_rppa_solid_samples.gct
- processed_solid_samples/depmap23q2_crispr_solid_samples.gct
"""

import argparse
import numpy as np
import pandas as pd
import sys
from typing import Dict
import unittest

import src.GeneSetRefinement as gsr


class ExpressionTests(unittest.TestCase):
	expr: gsr.Expression
	rng: np.random.Generator

	def setUp(self):
		depmap_path = sys.argv[1]

		if depmap_path[-1] != '/':
			depmap_path += '/'

		expression_path = (
			f"{depmap_path}/processed_solid_samples/"
			f"depmap23q2_expression_solid_samples.gct"
		)

		self.expr = gsr.Expression.from_gct(
			expression_path,
			min_counts = -100
		)

		self.expr.normalize()

		self.rng = np.random.default_rng(49)

		rnd_genes = self.rng.choice(
			self.expr.gene_names,
			size = 100,
			replace = False,
		).tolist()

		self.expr = gsr.Data2D.subset(
			self.expr, 
			row_names = rnd_genes,
		)

	def test_shape(self):
		self.assertEqual(self.expr.n_genes, 100)
		self.assertEqual(self.expr.n_samples, 1122)

	def test_subset(self):
		subs_samples = ["ACH-000696", "ACH-000885"]

		subs = gsr.Data2D.subset(
			self.expr,
			column_names = subs_samples
		)

		self.assertEqual(subs.n_genes, self.expr.n_genes)
		self.assertEqual(subs.n_samples, len(subs_samples))

	def test_subset_keep(self):
		keep = self.expr.subset_random_samples(
			0.3,
			self.rng,
			return_both = False,
		)

		self.assertIsInstance(keep, gsr.Expression)
		self.assertEqual(keep.n_samples, 336) # type: ignore

	def test_subset_keep_disc(self):
		subs_res = self.expr.subset_random_samples(
			0.3,
			self.rng,
			return_both = True
		)

		self.assertIsInstance(subs_res, tuple)
		keep, disc = subs_res # type: ignore

		self.assertEqual(keep.n_samples, 336)
		self.assertEqual(disc.n_samples, 786)

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

	def setUp(self):
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

		self.phenotypes = gsr.Phenotypes.from_gcts(paths_d)

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

	def setUp(self):
		self.gene_set_path = (
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

	def setUp(self):
		depmap_path = sys.argv[1]

		if depmap_path[-1] != '/':
			depmap_path += '/'

		self.expression_path = (
			f"{depmap_path}/processed_solid_samples/"
			f"depmap23q2_expression_solid_samples.gct"
		)

		self.rppa_path = (
			f"{depmap_path}/processed_solid_samples/"
			f"depmap23q2_rppa_solid_samples.gct"
		)

		self.proteomics_path = (
			f"{depmap_path}/processed_solid_samples/"
			f"depmap23q2_proteomics_solid_samples.gct"
		)

		self.paths_d = {
			"rppa": self.rppa_path,
			"proteomics": self.proteomics_path
		}

		self.gene_set_path = (
			"examples/input_gene_sets/REACTOME_SIGNALING_BY_ERBB2_v6.0.gmt"
		)

		self.gene_set_name = "REACTOME_SIGNALING_BY_ERBB2_v6.0"

		self.ref = gsr.Refinement(
			self.expression_path,
			self.paths_d,
			self.gene_set_path,
			"REACTOME_SIGNALING_BY_ERBB2_v6.0",
			[2, 3, 4],
			verbose = True
		)

		self.training_expr = self.ref._expr.subset_random_samples(
			0.67,
			self.ref._rng,
			return_both = False
		)

		self.ii = gsr.InnerIteration(
			self.training_expr,
			self.ref._gs,
			0,
			3,
			self.ref._rng
		)

		self.ii.run()

	def test_instantiating_refinement(self):
		self.assertListEqual(self.ref.k_values, [2, 3, 4])

	def test_inner_iteration_instantiation(self):
		self.assertEqual(
			self.ii._gen_expr.n_samples, 
			int(self.training_expr.n_samples * 0.50)
		)

	def test_inner_iteration_get_A(self):
		self.assertEqual(
			self.ii._A.n_genes,
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
			gsr.Data2D.subset(
				self.ii.A,
				["CDKN2A", "ERBB2"], 
				["asdf", "asfasf"]
			)

	def test_good_A_matrix_subset(self):
		subs_a = gsr.Data2D.subset(
			self.ii.A,
			["CDKN2A", "ERBB2"],
			[]
		)

		self.assertIsInstance(subs_a, gsr.A_Matrix)

	def test_inner_iteration_nmf(self):
		self.assertListEqual(
			list(self.ii.W.shape),
			[self.ii.A.n_genes, self.ii.k]
		)

		self.assertListEqual(
			list(self.ii.H.shape),
			[self.ii.k, self.ii.A.n_samples]
		)

	def test_W_matrix_good_subset(self):
		subs_w = gsr.Data2D.subset(
			self.ii.W,
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
			gsr.Data2D.subset(
				self.ii.W,
				["CDKN2A", "ERBB2"],
				["asfd", "adfsdf"]
			)

	def test_gene_comp_ic_shape(self):
		self.assertEqual(
			self.ii._gene_comp_ic.shape[0], 
			self.ii.A.shape[0]
		)

		self.assertEqual(
			self.ii._gene_comp_ic.shape[1],
			self.ii.H.shape[0]
		)


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

	def setUp(self):
		self.test_data: pd.DataFrame = pd.DataFrame(
			{"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
			index = ["r1", "r2", "r3"]
		)

		self.good_obj = self.GoodRealData(self.test_data)
		self.bad_obj = self.BadRealData()
		self.new_attr_obj = self.DataNewAttr(self.test_data, param = 5)

	def test_check_attrs(self):
		with self.assertRaisesRegex(	
			AttributeError,
			(
				r"Missing attribute\(s\) _data. Does BadRealData\.__init__\(\) "
				r"call super\(\)\.__init__\(\)\?"
			)
		):
			self.bad_obj.data

		self.assertEqual(
			self.good_obj.row_names,
			["r1", "r2", "r3"]
		)

	def test_parent_attrs(self):
		self.assertSetEqual(
			set(self.good_obj._base_attrs.keys()),
			set(["_data", "_base_attrs"])
		)

	def test_child_attrs(self):
		self.assertDictEqual(
			self.good_obj._get_child_attrs(),
			{}
		)

		self.assertDictEqual(
			self.new_attr_obj._get_child_attrs(),
			{"_param": 5}
		)

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

		new_child_obj = self.new_attr_obj[["r1", "r3", "r4"], ["a", "c"]]

		self.assertDictEqual(
			self.new_attr_obj._get_child_attrs(),
			new_child_obj._get_child_attrs()
		)


if __name__ == "__main__":
	## https://stackoverflow.com/questions/2812218/
	## problem-with-sys-argv1-when-unittest-module-is-in-a-script
	unittest.main(
		argv = [sys.argv[0]],
		verbosity = 2,
		#defaultTest = ["RefinementTests.test_W_matrix_good_subset"]
	)





