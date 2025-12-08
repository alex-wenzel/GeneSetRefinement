from pathlib import Path
import sys
from typing import List

sys.path.insert(0, "../")

from src.GeneSetRefinement.Refinement import Refinement, kRefinement
from src.GeneSetRefinement.GeneSet import MSigDBQuery

if __name__ == "__main__":
	try:
		depmap_path_str = sys.argv[1]
	except IndexError:
		raise RuntimeError(
			(
				"Usage: python generate_reactome-erbb2-60.py "
				"/path/to/depmap_release/"
			)
		)
	
	depmap_path: Path = Path(depmap_path_str)

	depmap_version = depmap_path.parts[-1].lower()

	expr_path = depmap_path / (
		f"processed_solid_samples/"
		f"depmap{depmap_version}_solid_expression.gct"
	)

	rppa_path = depmap_path / (
		f"processed_solid_samples/"
		f"depmap{depmap_version}_solid_protein-rppa.gct"
	)
	phens_d = {
		"rppa": str(rppa_path),
	}

	input_set: List[MSigDBQuery] = [{
		"collection": "c2.cp.reactome",
		"release": "2025.1.Hs",
		"gene_set_name": "REACTOME_SIGNALING_BY_ERBB2"
	}]

	ref = Refinement(
		Path("example_results"),
		"erbb2-60",
		expr_path,
		phens_d,
		input_set,
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

	ref.preprocess()

	ref.run()
	ref.save()
	