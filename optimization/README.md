# Optimization Scripts

This folder contains multiple runnable examples. Use a `uv` virtual environment to keep
dependencies isolated.

## 1) Create and activate the uv environment

```sh
uv venv --python 3.13
source .venv/bin/activate
```

## 2) Install dependencies

```sh
uv pip install -r requirements.txt
```

## 3) Run the scripts

### ILP with explainability (PuLP)
```sh
python optimization.py
```

### Maximum entropy (continuous, CVXPY)
```sh
uv pip install -r requirements-cvxpy.txt
python optimization_maxent.py
```

### Hybrid objective (productivity + entropy, continuous, CVXPY)
```sh
uv pip install -r requirements-cvxpy.txt
python optimization_hybrid_entropy.py
```

## 4) Output figures

Each script saves figures under:

```
optimization/figs/<script_name>/<timestamp>/
```

Examples:
- `optimization/figs/optimization/20260118_170103/`
- `optimization/figs/optimization_maxent/20260118_170103/`
- `optimization/figs/optimization_hybrid_entropy/20260118_170103/`

## Notes

- `requirements.txt` is the ILP-only stack that installs cleanly on Python 3.13.
- `requirements-cvxpy.txt` adds CVXPY, which may require Python 3.11 or conda if wheels are unavailable for your platform.
