import os
import random

import pandas as pd
import pytest

HAM = ["are we meeting {w} tomorrow", "can you call me when you get home {w}", "thanks for dinner {w} see you soon",
       "i will be late {w} the traffic is bad", "did you finish the report {w} yet", "happy birthday {w} hope it is great"]
SPAM = ["WINNER claim your free prize {w} call 0906170{n} now", "URGENT your account is locked verify at http://secure{n}.xyz {w}",
        "free entry win cash {w} text WIN to 8{n} now", "congratulations you won £{n} {w} txt CLAIM to 80086"]
WORDS = "apple river stone cloud tiger music paper green happy quick lemon bridge window garden silver rocket".split()


def make_dataset(n=700, seed=0) -> pd.DataFrame:
    rng, rows = random.Random(seed), []
    for i in range(n):
        spam = rng.random() < 0.2
        tpl = rng.choice(SPAM if spam else HAM)
        rows.append((int(spam), tpl.format(w=f"{rng.choice(WORDS)}{i}", n=rng.randint(1000, 99999))))
    return pd.DataFrame(rows, columns=["label", "text"])


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory):
    p = tmp_path_factory.mktemp("data") / "synthetic.csv"
    make_dataset().rename(columns={"label": "v1", "text": "v2"}).assign(
        v1=lambda d: d["v1"].map({0: "ham", 1: "spam"})).to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def trained_home(tmp_path_factory, synthetic_csv):
    """A real (tiny) end-to-end training run in an isolated SPAMGUARD_HOME."""
    home = tmp_path_factory.mktemp("home")
    old = os.environ.get("SPAMGUARD_HOME"); os.environ["SPAMGUARD_HOME"] = str(home)
    from spamguard.config import load_params
    from spamguard.train import run
    params = load_params(overrides={"data": {"source": "local", "local_path": str(synthetic_csv)},
                                    "features": {"min_df": 1}, "training": {"n_jobs": 1}})
    summary = run(params, fast=True, models=["dummy_prior", "multinomial_nb", "complement_nb", "logistic_regression"],
                  use_mlflow=False)
    yield home, summary
    if old is None: os.environ.pop("SPAMGUARD_HOME", None)
    else: os.environ["SPAMGUARD_HOME"] = old


@pytest.fixture()
def predictor(trained_home):
    from spamguard.predictor import Predictor
    return Predictor.from_champion(log_predictions=True)
