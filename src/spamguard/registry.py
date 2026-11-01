"""File-based model registry: immutable, checksummed, versioned bundles + a champion pointer.

A *bundle* contains the **whole** sklearn Pipeline (vectoriser + classifier), so a vectoriser can
never be paired with the wrong model version. ``champion.json`` is the only mutable file and is
written atomically, which makes promotion / rollback a one-line operation.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from . import __version__
from .config import get_paths


def _atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=3).stdout.strip() or "nogit"
    except Exception:  # noqa: BLE001
        return "nogit"


@dataclass
class Bundle:
    version: str
    pipeline: object
    threshold: float
    metadata: dict


def new_version(data_fingerprint: str) -> str:
    return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{data_fingerprint[:6]}"


def save_bundle(pipeline, metadata: dict, models_dir: Path | None = None, promote: bool = True) -> str:
    models_dir = models_dir or get_paths().models
    version = metadata.get("version") or new_version(metadata.get("data_fingerprint", "000000"))
    vdir = models_dir / "versions" / version
    vdir.mkdir(parents=True, exist_ok=True)
    model_path = vdir / "model.joblib"
    joblib.dump(pipeline, model_path, compress=3)
    metadata = {**metadata, "version": version, "created_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": sha256_file(model_path), "git_sha": git_sha(), "spamguard_version": __version__,
                "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
                "model_file": "model.joblib"}
    _atomic_write(vdir / "metadata.json", json.dumps(metadata, indent=2, default=str))
    reg_path = models_dir / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {"versions": []}
    reg["versions"].append({"version": version, "created_utc": metadata["created_utc"],
                            "model_name": metadata.get("model_name"), "threshold": metadata.get("threshold"),
                            "pr_auc_test": metadata.get("metrics_test", {}).get("pr_auc")})
    _atomic_write(reg_path, json.dumps(reg, indent=2))
    if promote:
        promote_version(version, models_dir)
    return version


def promote_version(version: str, models_dir: Path | None = None):
    """Point the champion at ``version`` (also used for rollback)."""
    models_dir = models_dir or get_paths().models
    if not (models_dir / "versions" / version / "model.joblib").exists():
        raise FileNotFoundError(f"unknown version {version}")
    _atomic_write(models_dir / "champion.json",
                  json.dumps({"version": version, "promoted_utc": datetime.now(timezone.utc).isoformat()}))


def list_versions(models_dir: Path | None = None) -> list[dict]:
    p = (models_dir or get_paths().models) / "registry.json"
    return json.loads(p.read_text())["versions"] if p.exists() else []


def load_bundle(version: str | None = None, models_dir: Path | None = None, verify: bool = True) -> Bundle:
    models_dir = models_dir or get_paths().models
    if version is None:
        ptr = models_dir / "champion.json"
        if not ptr.exists():
            raise FileNotFoundError("No champion model yet - run `python -m spamguard.train` first.")
        version = json.loads(ptr.read_text())["version"]
    vdir = models_dir / "versions" / version
    meta = json.loads((vdir / "metadata.json").read_text())
    model_path = vdir / meta.get("model_file", "model.joblib")
    if verify and sha256_file(model_path) != meta["sha256"]:
        raise RuntimeError(f"Checksum mismatch for model {version}: file corrupted or tampered with.")
    return Bundle(version, joblib.load(model_path), float(meta["threshold"]), meta)
