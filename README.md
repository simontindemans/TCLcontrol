# TCLcontrol
Decentralised distribution-referred TCL controller

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4114089.svg)](https://doi.org/10.5281/zenodo.4114089)

[![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://dx.doi.org/10.24433/CO.4712510.v1)

This code accompanies the paper *Low-complexity algorithm for decentralized aggregate load control of thermostatic loads*,
to appear in IEEE Transactions on Industry Applications.

If you use (parts of) this code, please cite the published paper.

## Organization
Code is available in the directory `/code`. The Jupyter notebook `TCLcontrol_notebook.ipynb` reproduces all figures from the paper. These are available in `/results`.

## Dependencies
This code requires installation of the following libraries: `numpy`, `scipy`, `numba`, `matplotlib` and `statsmodels`. Only `numba` is required for the controller module.
