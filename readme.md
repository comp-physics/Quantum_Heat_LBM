# Quantum Heat LBM

**Quantum lattice Boltzmann algorithm for heat transfer with phase change**

This repo contains a minimal reference implementation of a **quantum lattice Boltzmann (QLBM)** solver for the 1D Stefan problem (heat transfer with phase change), as described in the accompanying manuscript.

## Repo layout

```
.
├── QLBM_phase_change.py   # Main simulation script (QLBM with phase change)
├── plot_maker.py          # Plotting/figure generation from saved results
└── Data/                  # Input/output data (created/used by the scripts)
```

## Quick start

1. Install deps (Python 3.9+ recommended):

```bash
pip install qiskit numpy scipy matplotlib
```

2. Run the QLBM simulation:

```bash
python QLBM_phase_change.py
```

This runs the 1D phase-change test (Stefan problem) and saves results in `Data/` (e.g., arrays/npz files and/or figures, depending on the script’s settings).

3. Make plots:

```bash
python plot_maker.py
```

Generates figures from whatever the simulation saved into `Data/`.

> Tip: If you want to change lattice size, material parameters (α, cp, L\_melt), boundary temperatures, or number of time steps, look for the parameter block at the top of `QLBM_phase_change.py` and edit as needed.

## What this code demonstrates

* Encoding of LBM distribution functions into qubit amplitudes for a D1Q3 lattice.
* Unitary **collision** and **streaming** steps arranged as a circuit.
* An interface-tracking strategy to handle the nonlinear enthalpy–temperature coupling at melting without frequent full-state collapses.
* Comparison-ready outputs (temperature profiles, interface position) to validate against analytic/classical baselines.

## Reproducibility

The default configuration reproduces the paper’s 1D melting test (with a modest lattice and time horizon suitable for simulator runs).
After `QLBM_phase_change.py` finishes, run `plot_maker.py` to render temperature evolution and interface tracking plots from the data in `Data/`.

## Citation

If this code helps your work, please cite the paper:

> C. Jawetz, Z. Song, S. H. Bryngelson, A. Alexeev, *Quantum lattice Boltzmann algorithm for heat transfer with phase change*, 2025
