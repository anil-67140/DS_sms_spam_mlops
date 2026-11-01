"""Data acquisition (Kaggle), cleaning, de-duplication and stratified splitting."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import PROJECT_ROOT, get_paths, load_params
from .logging_utils import get_logger

log = get_logger("spamguard.data")

DATASETS = {
    # Classic UCI SMS Spam Collection (5,574 msgs). Highest-quality labels -> default benchmark.
    "uci": {"slug": "uciml/sms-spam-collection-dataset", "dir": "uci"},
    # UCI set machine-translated into 21 languages (~100k rows). NOTE: translations of the same
    # message share a label *and* meaning, so a random split leaks. Use for multilingual
    # experiments only; see README "Datasets".
    "multilingual": {"slug": "dbarbedillo/sms-spam-multilingual-collection-dataset", "dir": "multilingual"},
}
LABEL_COLS = ["label", "labels", "v1", "category", "target", "class", "type", "spam"]
TEXT_COLS = ["text", "message", "v2", "sms", "body", "content", "email", "text_translated"]
LANG_COLS = ["lang", "language", "lang_code"]
LABEL_MAP = {"ham": 0, "legitimate": 0, "legit": 0, "not spam": 0, "0": 0, "0.0": 0,
             "spam": 1, "smish": 1, "smishing": 1, "phishing": 1, "1": 1, "1.0": 1}


# --------------------------------------------------------------------------- download
def download_kaggle(slug: str, dest: Path, force: bool = False) -> Path:
    """Download + unzip a Kaggle dataset straight into ``dest`` (idempotent; cached on Drive)."""
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".download_complete"
    if marker.exists() and not force:
        log.info("Dataset already cached at %s", dest)
        return dest
    if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_API_TOKEN")
            or (Path.home() / ".kaggle" / "kaggle.json").exists()):
        raise RuntimeError("Kaggle credentials not found. Set KAGGLE_USERNAME/KAGGLE_KEY "
                           "(or KAGGLE_API_TOKEN), or place ~/.kaggle/kaggle.json.")
    log.info("Downloading %s from Kaggle ...", slug)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Kaggle download failed: {res.stderr.strip() or res.stdout.strip()}")
    marker.write_text(slug)
    return dest


# --------------------------------------------------------------------------- loading
def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")


def _pick(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame[text, label(0/1), (lang)] regardless of the source's column naming."""
    lab, txt = _pick(list(df.columns), LABEL_COLS), _pick(list(df.columns), TEXT_COLS)
    if lab is None or txt is None:
        raise ValueError(f"Could not detect label/text columns in {list(df.columns)}")
    out = pd.DataFrame({"text": df[txt], "label": df[lab].astype(str).str.strip().str.lower().map(LABEL_MAP)})
    lang = _pick(list(df.columns), LANG_COLS)
    if lang:
        out["lang"] = df[lang]
    unknown = out["label"].isna().sum()
    if unknown:
        log.warning("Dropping %d rows with unrecognised labels", unknown)
    return out.dropna(subset=["label"]).astype({"label": int})


def load_raw(params: dict) -> pd.DataFrame:
    d = params["data"]
    paths = get_paths()
    if d["source"] == "local" or d.get("local_path"):
        path = Path(d["local_path"]).expanduser()
        log.info("Loading local file %s", path)
        return standardize(_read_csv(path))
    cfg = DATASETS[d["source"]]
    folder = paths.raw / cfg["dir"]
    try:
        download_kaggle(cfg["slug"], folder)
    except Exception as exc:  # noqa: BLE001
        if d.get("fallback_to_sample") and d["source"] == "uci":
            log.warning("%s -> falling back to bundled sample (spam.csv)", exc)
            return standardize(_read_csv(PROJECT_ROOT / "data" / "sample" / "spam.csv"))
        raise
    csvs = sorted(folder.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {folder}")
    log.info("Reading %s", csvs[0].name)
    return standardize(_read_csv(csvs[0]))


# --------------------------------------------------------------------------- cleaning
def clean(df: pd.DataFrame, max_chars: int = 1000) -> tuple[pd.DataFrame, dict]:
    """Strip, drop empties/outliers, de-duplicate on normalised text and drop label conflicts."""
    n0 = len(df)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[(df["text"].str.len() > 0) & (df["text"].str.len() <= max_chars)]
    key = df["text"].str.lower().str.replace(r"\s+", " ", regex=True)
    conflict = df.assign(_k=key).groupby("_k")["label"].nunique()
    conflict_keys = set(conflict[conflict > 1].index)
    df = df[~key.isin(conflict_keys)]
    before_dedup = len(df)
    df = df.assign(_k=key[df.index]).drop_duplicates("_k").drop(columns="_k").reset_index(drop=True)
    report = {"rows_raw": n0, "rows_after_cleaning": len(df),
              "duplicates_removed": before_dedup - len(df),
              "conflicting_label_texts_removed": len(conflict_keys),
              "spam_rate": round(float(df["label"].mean()), 4)}
    log.info("Cleaning report: %s", report)
    return df, report


# --------------------------------------------------------------------------- splitting
def stratified_split(df: pd.DataFrame, val_size: float, test_size: float, seed: int):
    """Stratified train/val/test. Test is touched exactly once at the very end."""
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size),
                                  stratify=train_val["label"], random_state=seed)
    return (train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for t, y in zip(df["text"].tolist(), df["label"].tolist()):
        h.update(f"{y}\t{t}\n".encode())
    return h.hexdigest()[:16]


def prepare(params: dict | None = None) -> dict:
    """Run download -> clean -> split; persist CSVs + manifest; return splits and manifest."""
    params = params or load_params()
    paths = get_paths()
    df, report = clean(load_raw(params), params["data"]["max_chars"])
    train, val, test = stratified_split(df, params["data"]["val_size"], params["data"]["test_size"],
                                        params["seed"])
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(paths.processed / f"{name}.csv", index=False)
    manifest = {
        "source": params["data"]["source"], "fingerprint": dataframe_fingerprint(df), **report,
        "splits": {n: {"rows": len(p), "spam_rate": round(float(p["label"].mean()), 4)}
                   for n, p in (("train", train), ("val", val), ("test", test))},
        "seed": params["seed"],
    }
    (paths.processed / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Splits: %s", manifest["splits"])
    return {"train": train, "val": val, "test": test, "manifest": manifest}


def load_splits() -> dict:
    paths = get_paths()
    return {n: pd.read_csv(paths.processed / f"{n}.csv") for n in ("train", "val", "test")}


if __name__ == "__main__":  # python -m spamguard.data  (DVC "prepare" stage)
    prepare()
