"""
This script is modified based on sklearn's RFE and RFECV implementation to support named features and SHAP Explainer.
"""

"""
BSD 3-Clause License

Copyright (c) 2007-2022 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
import numpy as np
import numbers
from sklearn.base import clone
from operator import attrgetter
from sklearn.utils import safe_sqr
from sklearn.model_selection import check_cv
from sklearn.metrics import check_scoring
from joblib import Parallel, effective_n_jobs
from sklearn.utils.fixes import delayed
from sklearn.base import is_classifier
from sklearn.utils.metaestimators import _safe_split
from sklearn.model_selection._validation import _score


def _rfe_single_fit(rfe, estimator, feature_names, X, y, train, test, scorer):
    """
    Return the score for a fit across one fold.
    """

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    func = lambda estimator, features: _score(
        estimator,
        rfe.get_X_wrap(X_test, features),
        y_test,
        scorer,
    )
    return rfe._fit(
        pd.DataFrame(X_train, columns=feature_names, index=np.arange(X_train.shape[0])),
        y_train,
        func,
    ).scores_


class ExtendRFE(RFE):
    def __init__(self, feature_names=None, *args, **kwargs):
        super(ExtendRFE, self).__init__(*args, **kwargs)
        self.feature_names = np.array(feature_names)

    def get_X_wrap(self, X, features):
        X_wrap = (
            pd.DataFrame(
                data=X[:, features],
                columns=self.feature_names[features],
                index=np.arange(X.shape[0]),
            )
            if self.feature_names is not None
            else X[:, features]
        )
        return X_wrap

    def _fit(self, X, y, step_score=None, **fit_params):
        # Parameter step_score controls the calculation of self.scores_
        # step_score is not exposed to users
        # and is used when implementing RFECV
        # self.scores_ will not be calculated when calling _fit through fit
        tags = self._get_tags()
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csc",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
            multi_output=True,
        )
        error_msg = (
            "n_features_to_select must be either None, a "
            "positive integer representing the absolute "
            "number of features or a float in (0.0, 1.0] "
            "representing a percentage of features to "
            f"select. Got {self.n_features_to_select}"
        )

        # Initialization
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        elif self.n_features_to_select < 0:
            raise ValueError(error_msg)
        elif isinstance(self.n_features_to_select, numbers.Integral):  # int
            n_features_to_select = self.n_features_to_select
        elif self.n_features_to_select > 1.0:  # float > 1
            raise ValueError(error_msg)
        else:  # float
            n_features_to_select = int(n_features * self.n_features_to_select)

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")

        support_ = np.ones(n_features, dtype=bool)
        ranking_ = np.ones(n_features, dtype=int)

        if step_score:
            self.scores_ = []

        # Elimination
        while np.sum(support_) > n_features_to_select:
            # Remaining features
            features = np.arange(n_features)[support_]

            # Rank the remaining features
            estimator = clone(self.estimator)
            if self.verbose > 0:
                print(f"Fitting estimator with {np.sum(support_)} features.")

            estimator.fit(
                self.get_X_wrap(X, features),
                y,
                **fit_params,
            )

            # Get importance and rank them
            importances = _get_feature_importances(
                estimator,
                self.importance_getter,
                self.get_X_wrap(X, features),
                transform_func="square",
            )
            ranks = np.argsort(importances)

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            # Eliminate the worse features
            threshold = min(step, np.sum(support_) - n_features_to_select)

            # Compute step score on the previous selection iteration
            # because 'estimator' must use features
            # that have not been eliminated yet
            if step_score:
                self.scores_.append(step_score(estimator, features))
            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1

            if self.feature_names is not None and self.verbose > 0:
                print(
                    f"{self.feature_names[features[ranks][:threshold]]} is eliminated."
                )

        # Set final attributes
        features = np.arange(n_features)[support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self.get_X_wrap(X, features), y, **fit_params)

        # Compute step score when only n_features_to_select features left
        if step_score:
            self.scores_.append(step_score(self.estimator_, features))
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self


class ExtendRFECV(ExtendRFE, RFECV):
    def fit(self, X, y, groups=None):
        """Fit the RFE model and automatically tune the number of selected features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where ``n_samples`` is the number of samples and
            ``n_features`` is the total number of features.

        y : array-like of shape (n_samples,)
            Target values (integers for classification, real numbers for
            regression).

        groups : array-like of shape (n_samples,) or None, default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:``cv``
            instance (e.g., :class:``~sklearn.model_selection.GroupKFold``).

            .. versionadded:: 0.20

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.feature_names = (
            np.array(X.columns) if type(X) == pd.DataFrame else np.arange(X.shape[0])
        )
        tags = self._get_tags()
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
            multi_output=True,
        )

        # Initialization
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")

        # Build an RFE object, which will evaluate and score each possible
        # feature count, down to self.min_features_to_select
        rfe = ExtendRFE(
            estimator=self.estimator,
            n_features_to_select=self.min_features_to_select,
            importance_getter=self.importance_getter,
            step=self.step,
            verbose=self.verbose,
            feature_names=self.feature_names,
        )

        # Determine the number of subsets of features by fitting across
        # the train folds and choosing the "features_to_select" parameter
        # that gives the least averaged error across all folds.

        # Note that joblib raises a non-picklable error for bound methods
        # even if n_jobs is set to 1 with the default multiprocessing
        # backend.
        # This branching is done so that to
        # make sure that user code that sets n_jobs to 1
        # and provides bound methods as scorers is not broken with the
        # addition of n_jobs parameter in version 0.18.

        if effective_n_jobs(self.n_jobs) == 1:
            parallel, func = list, _rfe_single_fit
        else:
            parallel = Parallel(n_jobs=self.n_jobs)
            func = delayed(_rfe_single_fit)

        scores = parallel(
            func(rfe, self.estimator, self.feature_names, X, y, train, test, scorer)
            for train, test in cv.split(X, y, groups)
        )

        scores = np.array(scores)
        scores_sum = np.sum(scores, axis=0)
        scores_sum_rev = scores_sum[::-1]
        argmax_idx = len(scores_sum) - np.argmax(scores_sum_rev) - 1
        n_features_to_select = max(
            n_features - (argmax_idx * step), self.min_features_to_select
        )

        # Re-execute an elimination with best_k over the whole set
        rfe = ExtendRFE(
            estimator=self.estimator,
            n_features_to_select=n_features_to_select,
            step=self.step,
            importance_getter=self.importance_getter,
            verbose=self.verbose,
            feature_names=self.feature_names,
        )

        rfe.fit(X, y)

        # Set final attributes
        self.support_ = rfe.support_
        self.n_features_ = rfe.n_features_
        self.ranking_ = rfe.ranking_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(
            self.get_X_wrap(X, self.get_support(indices=True)),
            y,
        )

        # reverse to stay consistent with before
        scores_rev = scores[:, ::-1]
        self.cv_results_ = {}
        self.cv_results_["mean_test_score"] = np.mean(scores_rev, axis=0)
        self.cv_results_["std_test_score"] = np.std(scores_rev, axis=0)

        for i in range(scores.shape[0]):
            self.cv_results_[f"split{i}_test_score"] = scores_rev[i]

        return self


def _get_feature_importances(estimator, getter, X, transform_func=None, norm_order=1):
    if isinstance(getter, str):
        if getter == "auto":
            if hasattr(estimator, "coef_"):
                getter = attrgetter("coef_")
            elif hasattr(estimator, "feature_importances_"):
                getter = attrgetter("feature_importances_")
            else:
                raise ValueError(
                    "when ``importance_getter=='auto'`, the underlying "
                    f"estimator {estimator.__class__.__name__} should have "
                    "`coef_` or `feature_importances_` attribute. Either "
                    "pass a fitted estimator to feature selector or call fit "
                    "before calling transform."
                )
        else:
            getter = attrgetter(getter)
        importances = getter(estimator)
    elif not callable(getter):
        raise ValueError("`importance_getter` has to be a string or `callable`")
    else:
        importances = getter(estimator, X)

    if transform_func is None:
        return importances
    elif transform_func == "norm":
        if importances.ndim == 1:
            importances = np.abs(importances)
        else:
            importances = np.linalg.norm(importances, axis=0, ord=norm_order)
    elif transform_func == "square":
        if importances.ndim == 1:
            importances = safe_sqr(importances)
        else:
            importances = safe_sqr(importances).sum(axis=0)
    else:
        raise ValueError(
            "Valid values for `transform_func` are "
            + "None, 'norm' and 'square'. Those two "
            + "transformation are only supported now"
        )

    return importances
