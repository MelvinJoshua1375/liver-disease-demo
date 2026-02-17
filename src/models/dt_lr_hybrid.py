"""Decision-tree segmented logistic regression hybrid estimator."""

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


class DTSegmentedLR(BaseEstimator, ClassifierMixin):
    """Decision Tree segments + Logistic Regression hybrid.

    Fits a shallow decision tree to discover leaf-node segments, one-hot
    encodes the leaf assignments, concatenates with the original features,
    and fits a logistic regression on the augmented feature set.

    Proper sklearn estimator -- no data leakage when used inside a Pipeline
    with cross-validation.
    """

    def __init__(
        self,
        dt_max_depth: int = 3,
        dt_random_state: int = 42,
        lr_max_iter: int = 1000,
        lr_C: float = 1.0,
    ):
        self.dt_max_depth = dt_max_depth
        self.dt_random_state = dt_random_state
        self.lr_max_iter = lr_max_iter
        self.lr_C = lr_C

    def fit(self, X, y):
        X = self._validate_X(X)
        self.dt_ = DecisionTreeClassifier(
            max_depth=self.dt_max_depth,
            random_state=self.dt_random_state,
        )
        self.dt_.fit(X, y)

        leaf_ids = self.dt_.apply(X).reshape(-1, 1)
        self.leaf_encoder_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.leaf_encoder_.fit(leaf_ids)

        X_aug = self._augment(X)
        self.lr_ = LogisticRegression(
            max_iter=self.lr_max_iter,
            C=self.lr_C,
            solver="lbfgs",
        )
        self.lr_.fit(X_aug, y)
        self.classes_ = self.lr_.classes_
        return self

    def predict(self, X):
        check_is_fitted(self, ["dt_", "lr_", "leaf_encoder_"])
        X = self._validate_X(X)
        X_aug = self._augment(X)
        return self.lr_.predict(X_aug)

    def predict_proba(self, X):
        check_is_fitted(self, ["dt_", "lr_", "leaf_encoder_"])
        X = self._validate_X(X)
        X_aug = self._augment(X)
        return self.lr_.predict_proba(X_aug)

    def _augment(self, X):
        """Get DT leaf nodes, one-hot encode, and hstack with original."""
        leaf_ids = self.dt_.apply(X).reshape(-1, 1)
        leaf_ohe = self.leaf_encoder_.transform(leaf_ids)
        return np.hstack([X, leaf_ohe])

    @staticmethod
    def _validate_X(X):
        """Convert sparse matrices to dense arrays."""
        if issparse(X):
            return X.toarray()
        return np.asarray(X)
