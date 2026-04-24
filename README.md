Code accompanying the article: _Regional, functional and transcriptomic decoding of multidimensional brain structure alterations in obsessive-compulsive disorder_

This repository contains:
1. Method implementations in `mpm/`.
2. Synthetic examples in `examples/` that reproduce the analysis logic on toy data.


## Method modules

- `mpm/univariate.py`: Pearson correlation, linear regression, Freedman-Lane permutation method, and Fisher's method for p-value combination.
- `mpm/machine_learning.py`: nested cross-validation ML with permutation testing.
- `mpm/cca.py`: canonical correlation analysis.
- `mpm/pareto.py`: generalized Pareto tail fitting for permutation-derived p-values.


## Examples

- `examples/example_regional.py`: regional univariate analysis.
- `examples/example_correlation.py`: correlation among regional effect maps.
- `examples/example_convergence.py`: convergence of effect evidence across neuroimaging phenotypes within regions.
- `examples/example_machine_learning.py`: nested cross-validated prediction with permutation testing.
- `examples/example_network.py`: convergence of effect evidence across regions within functional networks.
- `examples/example_cca.py`: imaging transcriptomics CCA workflow.


## Installation

Clone the repository and install its dependencies:

```bash
git clone https://github.com/csleo95/Multi-phenotype-morphometry.git
cd Multi-phenotype-morphometry
pip install -r requirements.txt
```

Alternatively, if you want importable mpm modules from anywhere, clone the repository and install it in editable mode:

```bash
git clone https://github.com/csleo95/Multi-phenotype-morphometry.git
cd Multi-phenotype-morphometry
pip install -e .
```


## Citing
If you use the methods or code implemented here in your research, please cite the following paper:

Saraiva, L.C., Sato, J.R., Sebenius, I. et al. Regional, functional and transcriptomic decoding of multidimensional brain structure alterations in obsessive-compulsive disorder.
