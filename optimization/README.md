# Optimization Scripts

This folder contains multiple runnable examples. Use a `uv` virtual environment to keep
dependencies isolated.

## 1) Create and activate the uv environment

```sh
uv venv --python 3.10
source .venv/bin/activate
```

## 2) Install dependencies

```sh
uv pip install -r requirements.txt
```

## 3) Run the scripts

### ILP with explainability (PuLP)
```sh
python optimization_v2.py
```

### Maximum entropy (continuous, CVXPY)
```sh
python optimization_maxent.py
```

### Hybrid objective (productivity + entropy, continuous, CVXPY)
```sh
python optimization_hybrid_entropy.py
```

## 4) Output figures

Each script saves figures under:

```
optimization/figs/<script_name>/<timestamp>/
```

Examples:
- `optimization/figs/optimization_v2/20260118_170103/`
- `optimization/figs/optimization_maxent/20260118_170103/`
- `optimization/figs/optimization_hybrid_entropy/20260118_170103/`
