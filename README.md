# Fashion-MNIST linear probes

This project compares pixels, random CNN features, and trained CNN features on
low-label Fashion-MNIST probes. Experiment 1 decodes footwear and tops from a
ten-class CNN. Experiment 2 compares fine ten-class supervision with matched
footwear/non-footwear and tops/non-tops objectives.

## Setup

The reference runs used Python 3.10.20 and CPU execution. A GPU is optional;
`--device auto` selects CUDA or MPS when available. Reserve roughly 1 GB for the
dataset and full outputs.

From the project directory on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Fashion-MNIST is downloaded to `data/` automatically on the first run. The
scripts create the fixed splits and scale pixels to `[0,1]`; no separate
preprocessing command is required.

## Quick check

This one-epoch run verifies the installation and complete pipeline:

```powershell
python fashion_mnist_probes.py --quick --device cpu --output-dir results/quick_main
```

## Reproduce the paper experiments

Run these commands from the project directory. Use new output directories: the
low-label probe scripts intentionally reject nonempty destinations.

### Experiment 1: layer-wise decoding

```powershell
python fashion_mnist_probes.py --seed 42 --device cpu --output-dir results/reproduction/main
python fashion_mnist_sample_efficiency.py --source-dir results/reproduction/main --output-dir results/reproduction/main_probes --seed 42 --sample-sizes 10 20 30 50 80 110 150 200 --repeats 10 --controlled-dim 32 --projection-seeds 0 1 2 3 4 --device cpu --threads 1
```

### Experiment 2: training objectives

```powershell
python fashion_mnist_extension.py --coarse-target footwear --seed 42 --device cpu --threads 2 --output-dir results/reproduction/footwear
python fashion_mnist_extension_sample_efficiency.py --source-dir results/reproduction/footwear --source-seed 42 --output-dir results/reproduction/footwear_probes --seed 42 --sample-sizes 10 20 30 50 80 110 150 200 --repeats 10 --controlled-dim 32 --projection-seeds 0 1 2 3 4 --device cpu --threads 2
python fashion_mnist_extension.py --coarse-target tops --seed 42 --device cpu --threads 2 --output-dir results/reproduction/tops
python fashion_mnist_extension_sample_efficiency.py --source-dir results/reproduction/tops --source-seed 42 --output-dir results/reproduction/tops_probes --seed 42 --sample-sizes 10 20 30 50 80 110 150 200 --repeats 10 --controlled-dim 32 --projection-seeds 0 1 2 3 4 --device cpu --threads 2
```

The full CPU workflow can take over an hour. Progress is printed to the terminal.
Each results directory contains its configuration and status, raw and aggregate
metrics, fitted probes, split information, and generated figures.

To recreate the paper's four-panel figure from the included reference results:

```powershell
python paper/make_extension_four_panel_figure.py
```

This writes PDF and PNG versions to `paper/figures/`.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Use `python <script> --help` for all optional settings. Completed reference runs
are in `results/`; the Overleaf-ready source is in `paper/overleaf/`.
