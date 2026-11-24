import numpy as np
import pytest
from sklearn import datasets, preprocessing
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split

from pyGRNN import GRNN


@pytest.fixture
def diabetes_split():
    data = datasets.load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessing.minmax_scale(data.data),
        preprocessing.minmax_scale(data.target.reshape(-1, 1)),
        test_size=0.25,
        random_state=0,
    )
    return X_train, X_test, y_train.ravel(), y_test.ravel()


def test_predict_before_fit_raises(diabetes_split):
    _, X_test, _, _ = diabetes_split
    with pytest.raises(NotFittedError):
        GRNN().predict(X_test)


def test_isotropic_fit_predict_shapes_and_accuracy(diabetes_split):
    X_train, X_test, y_train, y_test = diabetes_split
    model = GRNN(calibration="None", sigma=0.4).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape
    # Sanity bound: a reasonable bandwidth should beat predicting the mean.
    assert MSE(y_test, y_pred) < MSE(y_test, np.full_like(y_test, y_train.mean()))


def test_anisotropic_gradient_search_fits_one_sigma_per_feature(diabetes_split):
    X_train, X_test, y_train, _ = diabetes_split
    model = GRNN(calibration="gradient_search").fit(X_train, y_train)
    assert model.sigma.shape == (X_train.shape[1],)
    y_pred = model.predict(X_test)
    assert np.isfinite(y_pred).all()


def test_warm_start_default_calibration_runs(diabetes_split):
    X_train, X_test, y_train, _ = diabetes_split
    model = GRNN(n_splits=3).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert np.isfinite(y_pred).all()


def test_refitting_the_same_instance_does_not_corrupt_bounds(diabetes_split):
    """Regression test: fit() used to overwrite self.bnds with a nested tuple
    that grew on every call, crashing scipy.optimize.minimize on a second fit
    (ValueError: too many values to unpack)."""
    X_train, _, y_train, _ = diabetes_split
    model = GRNN(calibration="gradient_search", bnds=(0, None))
    model.fit(X_train, y_train)
    model.fit(X_train, y_train)  # must not raise
    assert model.bnds == (0, None)


def test_clone_resets_fit_state(diabetes_split):
    """clone() must always hand back an unfitted estimator (sklearn's
    contract for cross-validation and GridSearchCV internals).

    Known limitation, not covered by this test: `sigma` is both the
    constructor parameter and, after fit(), the calibrated bandwidth (see
    the GRNN docstring), so clone() of an already-fitted instance carries
    the calibrated value over as the new instance's `sigma` instead of
    resetting it to the constructor default. Fixing that would mean
    splitting `sigma` into a parameter and a `sigma_` fitted attribute,
    which breaks the attribute every existing example in this repo (and
    downstream users) reads today, so it is left as is."""
    X_train, _, y_train, _ = diabetes_split
    model = GRNN(calibration="gradient_search").fit(X_train, y_train)
    fresh = clone(model)
    assert not hasattr(fresh, "is_fitted_")
