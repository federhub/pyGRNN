# pyGRNN
[![Tests](https://github.com/federhub/pyGRNN/actions/workflows/tests.yml/badge.svg)](https://github.com/federhub/pyGRNN/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/pyGRNN.svg)](https://pypi.org/project/pyGRNN/)

Python implementation of the General Regression Neural Network (GRNN, also known as the Nadaraya-Watson estimator), with a scikit-learn compatible API, plus a feature selection module built on isotropic and anisotropic GRNN.

The anisotropic feature selector implements the method described in:

> F. Amato, F. Guignard, P. Jacquet, M. Kanevski. **On Feature Selection Using Anisotropic General Regression Neural Network.** *ESANN 2020 proceedings*, pp. 411-416. [Full paper (PDF)](https://www.esann.org/sites/default/files/proceedings/2020/ES2020-16.pdf)

## Install

```sh
pip install pyGRNN
```

Requires Python 3.10 or newer. Dependencies (NumPy, SciPy, scikit-learn, pandas, matplotlib, seaborn) are installed automatically.

## Background

### GRNN

GRNN is the neural-network formulation of the Nadaraya-Watson kernel regression estimator. The prediction at a point `x` is a weighted average of all the observed targets `y_i`, with weights given by a kernel of the distance between `x` and each training point `x_i`:

```
ŷ(x) = Σ_i y_i K((x - x_i) / h) / Σ_i K((x - x_i) / h)
```

The only parameter to calibrate is the kernel bandwidth `h`, which makes GRNN fast to train compared with other feed-forward neural networks.

- **Isotropic GRNN (IGRNN)** uses a single bandwidth for all features.
- **Anisotropic GRNN (AGRNN)** uses a Gaussian kernel with one bandwidth `σ_j` per feature:

  ```
  K = exp( - Σ_j (x_j - x_ij)² / (2 σ_j²) )
  ```

  When `σ_j` grows, feature `j` stops contributing to the prediction; a small `σ_j` means the feature is highly discriminative. Calibrating the bandwidths (here by minimizing the cross-validated MSE with L-BFGS-B) therefore scales each feature by its explanatory power, which makes AGRNN an embedded feature selection method.

### Feature selection

Features can be completely irrelevant, weakly relevant and redundant, weakly relevant and non-redundant, or strongly relevant. pyGRNN provides two families of selectors:

- **`Isotropic_selector`** uses IGRNN as a wrapper (exhaustive, forward or backward search) to describe the whole input space: it identifies relevant, redundant and irrelevant features, and analyses the *relatedness* between features, i.e. how well each input can be predicted non-linearly from the others.
- **`Anisotropic_selector.max_dist()`** is the AGRNN-based Selector (AS) of the ESANN 2020 paper:
  1. rescale each feature to [0, 1];
  2. calibrate an AGRNN to find the optimal bandwidths `σ_j`;
  3. label feature `j` as relevant if `σ_j ≤ 1`. Above 1, the kernel gives almost the same small weight to every training point along that feature, so the feature's contribution to the regression becomes negligible.

  All features are evaluated at once, including their non-linear interactions, and the bandwidth values also rank the features by importance. Redundant features that carry the same information as relevant ones can also fall below the threshold (in the paper's butterfly benchmark, `J4 = X1² - X2²` is selected together with the relevant `X1` and `X2`).

For a step-by-step introduction, see the [Jupyter tutorial](https://github.com/federhub/pyGRNN/blob/master/Tutorial/Tutorial_PyGRNN.ipynb) and the [slides](https://github.com/federhub/pyGRNN/blob/master/Feature_Selection_with_GRNN.pdf).

## Examples

### Regression with isotropic and anisotropic GRNN

```py
import numpy as np
from sklearn import datasets, preprocessing
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV, train_test_split

from pyGRNN import GRNN

# Load the diabetes dataset and rescale features and target to [0, 1]
X, y = datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    preprocessing.minmax_scale(X),
    preprocessing.minmax_scale(y.reshape(-1, 1)).ravel(),
    test_size=0.25,
    random_state=42,
)

# Example 1: isotropic GRNN, bandwidth chosen by grid-search cross-validation
params = {"sigma": list(np.arange(0.1, 4, 0.01)), "calibration": ["None"]}
grid = GridSearchCV(GRNN(), param_grid=params, scoring="neg_mean_squared_error", cv=5)
grid.fit(X_train, y_train)
mse_IGRNN = MSE(y_test, grid.best_estimator_.predict(X_test))

# Example 2: anisotropic GRNN, one bandwidth per feature calibrated with L-BFGS-B
AGRNN = GRNN(calibration="gradient_search")
AGRNN.fit(X_train, y_train)
sigma = AGRNN.sigma  # calibrated bandwidths, one per feature
mse_AGRNN = MSE(y_test, AGRNN.predict(X_test))
```

`calibration` can be `"warm_start"` (default: first calibrate a single isotropic bandwidth, then use it as the starting point for the anisotropic search), `"gradient_search"` (anisotropic search starting from `sigma`), or `"None"` (use `sigma` as given).

### Feature selection

```py
from sklearn import datasets

from pyGRNN import feature_selection as FS

X, y = datasets.load_diabetes(return_X_y=True)
featnames = datasets.load_diabetes().feature_names

# Example 1: anisotropic selector (ESANN 2020 method) to keep only the relevant features
AS = FS.Anisotropic_selector()
AS.max_dist(X, y, feature_names=featnames)
AS.best_inSpaceNames  # names of the relevant features
AS.best_inSpace       # X restricted to those features (original scale)

# Example 2: explore data dependencies in the input space through the
# relatedness between features
IS = FS.Isotropic_selector(bandwidth="rule-of-thumb")
IS.relatidness(X, feature_names=featnames)
IS.plot_(feature_names=featnames)

# Example 3: exhaustive search for the best subset; a rule of thumb sets the
# bandwidth for each subset of features
IS = FS.Isotropic_selector(bandwidth="rule-of-thumb")
IS.es(X, y, feature_names=featnames)

# Example 4: complete analysis of the input space, recognising relevant,
# redundant and irrelevant features
IS = FS.Isotropic_selector(bandwidth="rule-of-thumb")
IS.feat_selection(X, y, feature_names=featnames, strategy="es")
```

`Isotropic_selector` also provides forward (`ffs`) and backward (`bfs`) search, and `Anisotropic_selector` a backward elimination (`bfe`) that repeatedly drops the feature with the largest bandwidth.

## Citation

If you use pyGRNN in your research, please cite:

```bibtex
@inproceedings{amato2020agrnn,
  title     = {On Feature Selection Using Anisotropic General Regression Neural Network},
  author    = {Amato, Federico and Guignard, Fabian and Jacquet, Philippe and Kanevski, Mikhail},
  booktitle = {ESANN 2020 proceedings, European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning},
  pages     = {411--416},
  year      = {2020},
  isbn      = {978-2-87587-074-2}
}
```

## What's new in 0.2.0

- `Anisotropic_selector.max_dist()` now follows the published algorithm: features are rescaled to [0, 1] internally and the relevance threshold is `σ ≤ 1` (previously no rescaling and `σ < sqrt(n_features)`). **Results differ from 0.1.x.** On data with very different feature ranges, the old version could fail to calibrate and select no features.
- Calling `fit()` twice on the same `GRNN` no longer crashes.
- `fit()` no longer silences NumPy divide/invalid warnings for the whole process.
- Several string comparisons that could take the wrong branch were fixed, and selector methods now consistently return `self`.
- Packaging moved to `pyproject.toml`; Python 3.10+ is required.

## Development

```sh
pip install -e ".[test]"
pytest
```

## Authors

* **Federico Amato**

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/federhub/pyGRNN/blob/master/LICENSE) file for details.
