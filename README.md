# GeneSetRefinement
A Python package implementing Gene Set Refinement. 

Quick start:

To run Gene Set Refinement with an initial gene set called `"gene set name"` in
the file `gene_set.gmt` with `k = 3` to `k = 9`

```python
import GeneSetRefinement as gsr

compendium_expression = "path/to/expression.gct"

phenotype_paths = {
	"data_type_1": "path/to/data1.gct",
	"data_type_2": "path/to/data2.gct",
}

input_gene_set_path = "path/to/gene_set.gmt"

refinement = gsr.Refinement(
	compendium_expression,
	phenotype_paths,
	input_gene_set_path,
	"gene set name",
	list(range(2, 10))
)

refinement.run()

## Print the gene sets for a value of `k`

for cluster_component in ref.component_clusters[k].values():
    print(cluster_component.gene_set.genes)

## I/O functions - save and use results later

refinement.save("refinement_result.pickle")

refinement = gsr.Refinement.load("refinement_result.pickle")
```
