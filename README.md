# Gaussian-Bump GPR and Inverse Design

Python code for the Gaussian-process surrogate modelling, validation and inverse-design calculations in *Reverse Design of Fluid-Solid Interfacial Bumps for Passive Heat Transfer Enhancement*.

## Prerequisites

- Python 3.11 or later
- NumPy, pandas, SciPy and scikit-learn

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The code was tested with Python 3.13.10 and scikit-learn 1.8.0.

## Overview

The single-bump model uses `eps_w` and `eps_h`. The H3 model uses `s_mm`, `alpha_w` and `alpha_h`. Four GPR models predict `Nu_ratio_avg`, `P_ratio`, `T_ratio_hot` (`Tmax/T0`) and `T_ratio_cold` (`T0/Tmin`). The temperature ratio and PEC are then calculated as

```text
Tmax/Tmin = T_ratio_hot * T_ratio_cold
PEC = Nu_ratio_avg / (P_ratio^(1/3) * (Tmax/Tmin)^0.1)
```

Main files:

- `surrogate.py`: GPR training and prediction
- `validation.py`: Group-CV validation and PEC reconstruction
- `optimization.py`: grid screening, L-BFGS-B and SLSQP
- `inverse_query.py`: direct feasible-set query and optional refinement
- `datasets.py`: CSV column standardisation

## Data

The data are supplied separately with the paper. Use Dataset 2 (`Single_bump_400_data.csv`) to train and validate the single-bump GPR models and Dataset 3 (`H3_720_data.csv`) for the H3 GPR models.

## Example

```python
from bump_inverse_design import load_csv, fit_surrogate_bundle

data, schema = load_csv("Dataset 2.csv", case="single")
X = data.loc[:, schema.feature_columns].to_numpy()
y = {name: data[name].to_numpy() for name in schema.target_columns}
models = fit_surrogate_bundle(X, y, case="single")
```

Run the included checks with

```bash
python -m unittest discover -s tests -v
python examples/minimal_smoke_test.py
```
