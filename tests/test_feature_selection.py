import numpy as np

from pyGRNN.feature_selection import Isotropic_selector


def _tiny_dataset(n_samples=20, n_features=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n_samples, n_features)
    y = X[:, 0] + 0.1 * rng.rand(n_samples)  # only feature 0 matters
    return X, y


def test_exhaustive_search_runs_and_ranks_features():
    X, y = _tiny_dataset()
    selector = Isotropic_selector(bandwidth="rule-of-thumb")
    selector.es(X, y, feature_names=["a", "b", "c"])
    assert selector.best_inSpaceNames  # found a non-empty best subset
    assert hasattr(selector, "best_model")
