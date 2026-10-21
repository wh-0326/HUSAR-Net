# HUSAR-Net

Hierarchical Uncertainty-Aware Spatial Allocation Network.

Minimal code release for the HUSAR objective, synthetic instance generation,
REINFORCE training, and greedy / sampling / beam-search inference.
The default network is the full N4 architecture; N3 is an ablation.

## Install

Tested with Python 3.12, NumPy 2.2.6, and PyTorch 2.11.0.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux / macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

A synthetic smoke test (random weights; no pretrained model is included):

```bash
python code/scripts/demo.py
```

Train the full N4 network from scratch (900 iterations per seed):

```bash
python code/scripts/train_husarnet.py --tag N4_full_s0 --seed 0 --iters 900 --lr 2e-4 --entropy-coef 0.02
python code/scripts/demo.py --checkpoint outputs/husarnet_N4_full_s0/best.pt
```

Training uses CUDA when available and otherwise falls back to CPU. Add
`--device cpu` to explicitly train on CPU. Repeat with seeds 0-4 for multiple runs.
`best.pt` is selected on fixed synthetic validation instances by the mean
objective ratio relative to the LCB greedy reference. For other inference modes,
add `--mode sample --n-samples 4` or `--mode beam --beam 4` to the demo.

## Files and scope

- `code/husarnet/`: objective, weights, environment, network, training, and decoders.
- `code/legacy/`: earlier baseline implementations from the same research project.
- `code/scripts/`: synthetic demo and training entry point.
- `code/tests/`: objective, environment, decoder, and checkpoint checks.

The core implementation comes from the source snapshot used for the 2026-10-21
results. Release changes make paths portable, remove the real-data loader, record
network settings in checkpoints, and fail explicitly if validation references fail.
Impossible synthetic hierarchy sizes are rejected. Checkpoint metadata use plain
Python scalars for weights-only loading. The objective and policy computations
are retained. The training CLI uses the
recorded learning rate, entropy coefficient, and single-thread setting.

Real company data, geographic files, trained weights, and paper result tables are
not included. This package supports synthetic method reproduction; it does not
reproduce every real-data experiment or reported paper table. MILP baselines
optionally require `gurobipy` and a valid Gurobi license.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest code/tests -q
```

MIT License; see [LICENSE](LICENSE).
