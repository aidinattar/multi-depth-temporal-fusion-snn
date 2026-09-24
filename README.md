# Multi-Depth Temporal Fusion utilizing residual- and consensus-inspired agreement feature routing

Code for the experiments in *Multi-Depth Temporal Fusion utilizing residual- and consensus-inspired agreement feature routing*.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[torchvision-data]'
```

## Data

MNIST, Fashion-MNIST, and CIFAR-10 can be downloaded automatically during
training. Convert a standard N-MNIST event directory once before use:

```bash
python scripts/prepare_nmnist_npz.py \
  --input-root /path/to/N-MNIST \
  --output data/N-MNIST.npz
```

## Training

Experiment configurations are available under `configs/experiments/`:

```text
full_model_mnist.yaml
full_model_fashion_mnist.yaml
full_model_cifar10.yaml
full_model_nmnist.yaml
```

Run an experiment with:

```bash
python scripts/train_full_model.py \
  --experiment configs/experiments/full_model_mnist.yaml \
  --dataset-root data \
  --download-data \
  --seed 0
```

For N-MNIST, select `full_model_nmnist.yaml` and omit `--download-data`.
Run artifacts and checkpoints are written under `runs/`.

## Tests

```bash
python -m pip install -e '.[test]'
pytest -q
```

## Citation
<!-- 
```bibtex
@article{attar2026multidepth,
  title         = {Multi-Depth Temporal Fusion utilizing residual- and consensus-inspired agreement feature routing},
  author        = {Attar, Aidin and Cicciarella, Eleonora and Rossi, Michele},
  journal       = {arXiv preprint arXiv:2609.XXXXX},
  year          = {2026},
  eprint        = {2609.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.NE}
}
``` -->
