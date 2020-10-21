# TCLcontrol
Decentralised distribution-referred TCL controller

# MLMC-PSCC2020
[![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://codeocean.com/capsule/0874781/tree)

This code accompanies the paper *Low-complexity control algorithm for decentralised demand response using thermostatic loads* by
Simon Tindemans and Goran Strbac, 2019 IEEE Conference on Environment and Electrical Engineering (EEEIC 2019), Genoa (Italy).
doi: [10.1109/EEEIC.2019.8783359](https://dx.doi.org/10.1109/EEEIC.2019.8783359).

A freely accessible preprint is available at: https://arxiv.org/abs/1904.12401. If you use (parts of) this code, please cite the published paper.

## Organization
Code is available in the directory `/code`. The Jupyter notebook `TCLcontrol_notebook.ipynb` reproduces all figures from the paper. These are available in `/results`.

## Dependencies
This code requires installation of the following libraries: `numpy`, `scipy`, `numba` and `matplotlib`. Only `numba` is required for the controller module.
