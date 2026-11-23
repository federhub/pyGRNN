import matplotlib

# feature_selection.py imports matplotlib.pyplot at module load time; force a
# non-interactive backend so tests run headless (CI has no display).
matplotlib.use("Agg")
