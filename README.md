# pytoda
[![build](https://github.com/PaccMann/paccmann_datasets/workflows/build/badge.svg)](https://github.com/PaccMann/paccmann_datasets/actions)
[![codecov](https://codecov.io/gh/PaccMann/paccmann_datasets/branch/master/graph/badge.svg?token=C10ICE7S0Q)](https://codecov.io/gh/PaccMann/paccmann_datasets)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code quality](https://www.code-inspector.com/project/22043/score/svg)](https://frontend.code-inspector.com/)
[![Code grade](https://www.code-inspector.com/project/22043/status/svg)](https://frontend.code-inspector.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

pytoda - PaccMann P*yTo*rch *Da*taset Classes

A python package that eases handling biochemical data for deep learning applications with pytorch.
Please find the full documentation [here](https://paccmann.github.io/paccmann_datasets/).

## Requirements

- `conda>=3.7`

## Installation

Create a conda environment:

```sh
conda env create -f conda.yml
```

Activate the environment:

```sh
conda activate pytoda
```

Install:

```sh
pip install .
```

### development

Create the `conda` environment as before, then install in editable mode for development:

```sh
pip install -r dev_requirements.txt
pip install --user --no-use-pep517 -e .
```

## Examples

For some examples on how to use `pytoda` see [here](./examples)

## References

If you use `pytoda` in your projects, please cite the following:

```bib
@article{born2021datadriven,
  author = {Born, Jannis and Manica, Matteo and Cadow, Joris and Markert, Greta and Mill, Nil Adell and Filipavicius, Modestas and Janakarajan, Nikita and Cardinale, Antonio and Laino, Teodoro and {Rodr{\'{i}}guez Mart{\'{i}}nez}, Mar{\'{i}}a},
  doi = {10.1088/2632-2153/abe808},
  issn = {2632-2153},
  journal = {Machine Learning: Science and Technology},
  number = {2},
  pages = {025024},
  title = {{Data-driven molecular design for discovery and synthesis of novel ligands: a case study on SARS-CoV-2}},
  url = {https://iopscience.iop.org/article/10.1088/2632-2153/abe808},
  volume = {2},
  year = {2021}
}
@article{born2021paccmannrl,
    title = {PaccMann$^{RL}$: De novo generation of hit-like anticancer molecules from transcriptomic data via reinforcement learning},
    journal = {iScience},
    volume = {24},
    number = {4},
    year = {2021},
    issn = {2589-0042},
    doi = {https://doi.org/10.1016/j.isci.2021.102269},
    url = {https://www.cell.com/iscience/fulltext/S2589-0042(21)00237-6},
    author = {Jannis Born and Matteo Manica and Ali Oskooei and Joris Cadow and Greta Markert and Mar{\'\i}a Rodr{\'\i}guez Mart{\'\i}nez}}
}
```
