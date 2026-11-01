"""Model zoo: 17 candidates (16 real models + a dummy sanity baseline) with search spaces."""
from __future__ import annotations

from dataclasses import dataclass, field

from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import build_features


@dataclass
class ModelSpec:
    name: str
    family: str
    estimator: BaseEstimator
    space: dict = field(default_factory=dict)
    heavy: bool = False
    selectable: bool = True   # dummy baseline is reported but never chosen as champion
    description: str = ""


def model_zoo(seed: int = 42, n_jobs: int = -1) -> dict[str, ModelSpec]:
    w = [None, "balanced"]
    z: dict[str, ModelSpec] = {}

    def add(spec: ModelSpec): z[spec.name] = spec

    add(ModelSpec("dummy_prior", "baseline", DummyClassifier(strategy="prior"),
                  selectable=False, description="Always predicts class prior (sanity floor)"))
    add(ModelSpec("multinomial_nb", "naive_bayes", MultinomialNB(),
                  {"clf__alpha": loguniform(1e-2, 2)}, description="Original notebook's winner"))
    add(ModelSpec("complement_nb", "naive_bayes", ComplementNB(),
                  {"clf__alpha": loguniform(1e-2, 2), "clf__norm": [True, False]}))
    add(ModelSpec("bernoulli_nb", "naive_bayes", BernoulliNB(),
                  {"clf__alpha": loguniform(1e-2, 2), "clf__binarize": [0.0, 0.05, 0.1]}))
    add(ModelSpec("logistic_regression", "linear",
                  LogisticRegression(solver="liblinear", max_iter=2000, random_state=seed),
                  {"clf__C": loguniform(0.1, 200), "clf__class_weight": w,
                   "clf__penalty": ["l1", "l2"]}))
    add(ModelSpec("sgd_modified_huber", "linear",
                  SGDClassifier(loss="modified_huber", max_iter=2000, tol=1e-4, random_state=seed),
                  {"clf__alpha": loguniform(1e-6, 1e-3), "clf__class_weight": w,
                   "clf__penalty": ["l2", "elasticnet"]}))
    add(ModelSpec("linear_svm", "svm",
                  CalibratedClassifierCV(LinearSVC(random_state=seed, max_iter=5000), cv=3),
                  {"clf__estimator__C": loguniform(0.05, 20), "clf__estimator__class_weight": w},
                  description="LinearSVC + sigmoid calibration (gives probabilities)"))
    add(ModelSpec("rbf_svm", "svm", SVC(kernel="rbf", probability=True, random_state=seed),
                  {"clf__C": loguniform(0.5, 50), "clf__gamma": loguniform(0.1, 3),
                   "clf__class_weight": w}, heavy=True))
    add(ModelSpec("random_forest", "tree_ensemble",
                  RandomForestClassifier(n_estimators=250, n_jobs=n_jobs, random_state=seed),
                  {"clf__max_depth": [None, 30, 60], "clf__min_samples_leaf": randint(1, 4),
                   "clf__max_features": ["sqrt", 0.05], "clf__class_weight": w}, heavy=True))
    add(ModelSpec("extra_trees", "tree_ensemble",
                  ExtraTreesClassifier(n_estimators=250, n_jobs=n_jobs, random_state=seed),
                  {"clf__max_depth": [None, 30, 60], "clf__min_samples_leaf": randint(1, 4),
                   "clf__class_weight": w}, heavy=True))
    add(ModelSpec("decision_tree", "tree",
                  DecisionTreeClassifier(random_state=seed),
                  {"clf__max_depth": randint(3, 25), "clf__min_samples_leaf": randint(1, 6),
                   "clf__class_weight": w}))
    add(ModelSpec("adaboost", "tree_ensemble",
                  AdaBoostClassifier(n_estimators=100, random_state=seed),
                  {"clf__learning_rate": loguniform(0.1, 1.5), "clf__n_estimators": [50, 100, 150]},
                  heavy=True))
    add(ModelSpec("knn_cosine", "instance",
                  KNeighborsClassifier(metric="cosine", algorithm="brute", n_jobs=n_jobs),
                  {"clf__n_neighbors": randint(3, 16), "clf__weights": ["uniform", "distance"]}))
    try:
        from lightgbm import LGBMClassifier
        add(ModelSpec("lightgbm", "boosting",
                      LGBMClassifier(n_estimators=300, verbose=-1, random_state=seed, n_jobs=n_jobs),
                      {"clf__learning_rate": loguniform(0.02, 0.2), "clf__num_leaves": randint(8, 48),
                       "clf__min_child_samples": randint(5, 30), "clf__colsample_bytree": uniform(0.3, 0.6),
                       "clf__subsample": uniform(0.6, 0.4), "clf__subsample_freq": [1]}, heavy=True))
    except ImportError:  # pragma: no cover
        pass
    try:
        from xgboost import XGBClassifier
        add(ModelSpec("xgboost", "boosting",
                      XGBClassifier(n_estimators=300, tree_method="hist", eval_metric="logloss",
                                    random_state=seed, n_jobs=n_jobs, verbosity=0),
                      {"clf__learning_rate": loguniform(0.02, 0.3), "clf__max_depth": randint(3, 9),
                       "clf__subsample": uniform(0.6, 0.4), "clf__colsample_bytree": uniform(0.3, 0.6)},
                      heavy=True))
    except ImportError:  # pragma: no cover
        pass

    # Ensembles (fixed, sensible hyper-parameters; members are themselves leakage-safe because
    # the whole thing sits behind the shared feature pipeline).
    base = [("nb", ComplementNB(alpha=0.3)),
            ("lr", LogisticRegression(solver="liblinear", C=20, random_state=seed)),
            ("svm", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=seed, max_iter=5000), cv=3))]
    add(ModelSpec("voting_soft", "ensemble", VotingClassifier(base, voting="soft"), heavy=True,
                  description="Soft vote of ComplementNB + LR + calibrated LinearSVM"))
    add(ModelSpec("stacking", "ensemble",
                  StackingClassifier(base, final_estimator=LogisticRegression(max_iter=1000),
                                     cv=5, stack_method="predict_proba"), heavy=True,
                  description="Stacked generalisation with LR meta-learner (5-fold OOF)"))
    return z


def build_pipeline(model_name: str, feat_cfg: dict, seed: int = 42, n_jobs: int = -1,
                   model_params: dict | None = None) -> Pipeline:
    spec = model_zoo(seed, n_jobs)[model_name]
    pipe = Pipeline([("features", build_features(feat_cfg)), ("clf", clone(spec.estimator))])
    if model_params:
        pipe.set_params(**model_params)
    return pipe
