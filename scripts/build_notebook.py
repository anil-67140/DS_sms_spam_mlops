"""Generates notebooks/SpamGuard_Colab.ipynb  (python scripts/build_notebook.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip("\n")))      # noqa: E731
M = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))  # noqa: E731

M("""
# 🛡️ SpamGuard — SMS Spam Detection (production-grade pipeline)
**Everything is saved to your Google Drive**, so a Colab disconnect never loses work:
dataset (downloaded from Kaggle), every trained model (checkpoints), metrics, plots, model registry, feedback, logs.
👉 **If the runtime disconnects, just use *Runtime ▸ Run all* again** — finished models are loaded from Drive and skipped.

### One-time setup (2 minutes)
1. Download **`sms-spam-mlops.zip`** and upload it to the root of **My Drive** (or drag it into Colab's 📁 Files panel).
2. Add your Kaggle credentials as **Colab Secrets** (🔑 icon in the left bar): `KAGGLE_USERNAME` and `KAGGLE_KEY`
   (Kaggle ▸ Settings ▸ API ▸ *Create New Token*). Turn **Notebook access** on for both. No Kaggle account? Set `LOCAL_CSV` below.
3. *Runtime ▸ Run all*. A CPU runtime is enough (GPU not needed). Full benchmark ≈ 10–25 min.
""")

M("## 1 · Mount Drive & unpack the project")
C('''
from google.colab import drive
drive.mount("/content/drive")

import os, sys, glob, shutil, zipfile
PROJECT_DIR = "/content/drive/MyDrive/sms-spam-mlops"      # <- ALL outputs live here (change if you like)
os.makedirs(PROJECT_DIR, exist_ok=True)

if not os.path.exists(f"{PROJECT_DIR}/src/spamguard/train.py"):
    candidates = ["/content/drive/MyDrive/sms-spam-mlops.zip", "/content/sms-spam-mlops.zip"]
    zip_path = next((p for p in candidates if os.path.exists(p)), None)
    if zip_path is None:
        from google.colab import files
        print("Upload sms-spam-mlops.zip ..."); zip_path = "/content/" + list(files.upload())[0]
    tmp = "/content/_unzip"; shutil.rmtree(tmp, ignore_errors=True)
    zipfile.ZipFile(zip_path).extractall(tmp)
    root = tmp if os.path.exists(f"{tmp}/src") else glob.glob(f"{tmp}/*/")[0]
    shutil.copytree(root, PROJECT_DIR, dirs_exist_ok=True)
    print("Project unpacked to", PROJECT_DIR)
else:
    print("Project already on Drive:", PROJECT_DIR)
''')

M("## 2 · Install dependencies")
C('''
!pip install -q -r {PROJECT_DIR}/requirements.txt
sys.path.insert(0, f"{PROJECT_DIR}/src")
os.environ["PYTHONPATH"] = f"{PROJECT_DIR}/src"
print("✅ dependencies installed (if Colab asks to restart the runtime, restart and Run all again)")
''')

M("## 3 · Kaggle credentials, dataset choice & settings")
C('''
# ---------------- settings you may change ----------------
DATASET   = "uci"    # "uci" (recommended benchmark, 5.5k msgs)  |  "multilingual" (~100k rows, 21 languages, see README caveat)
LOCAL_CSV = None     # e.g. "/content/drive/MyDrive/spam.csv" to skip Kaggle entirely
FAST      = False    # True = tiny search budget (3-4 min smoke test)
RESUME    = True     # True = reuse finished checkpoints from Drive
# ----------------------------------------------------------
os.environ["SPAMGUARD_HOME"] = PROJECT_DIR

if LOCAL_CSV is None:
    try:
        from google.colab import userdata
        os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
        print("✅ Kaggle credentials loaded from Colab Secrets")
    except Exception:
        from google.colab import files
        print("No Colab Secrets found - upload your kaggle.json"); files.upload()
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        shutil.move("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json")); os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

# MLflow's SQLite DB is kept on the fast local disk (Drive's FUSE layer can lock SQLite) and synced to Drive afterwards.
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:////content/mlflow.db"
if os.path.exists(f"{PROJECT_DIR}/mlflow.db") and not os.path.exists("/content/mlflow.db"):
    shutil.copy(f"{PROJECT_DIR}/mlflow.db", "/content/mlflow.db")
''')

M("""
## 4 · Train & benchmark **16 models** (leakage-safe, checkpointed)
Pipeline: clean → de-duplicate → **stratified** train/val/test → per model: random-search CV (PR-AUC) inside a
`Pipeline` (vectoriser fitted only on training folds) → champion chosen on **validation** → probability calibration →
decision threshold tuned on out-of-fold predictions → **one** final test evaluation with bootstrap CIs.
Each model is saved to `Drive/.../checkpoints/` the moment it finishes.
""")
C('''
from spamguard.config import load_params
from spamguard.train import run

over = {"data": {"source": "local" if LOCAL_CSV else DATASET, "local_path": LOCAL_CSV}, "training": {"n_jobs": -1}}
params = load_params(overrides=over)
summary = run(params, resume=RESUME, fast=FAST)
shutil.copy("/content/mlflow.db", f"{PROJECT_DIR}/mlflow.db") if os.path.exists("/content/mlflow.db") else None
print("\\nChampion:", summary["champion"], "| version:", summary["version"], "| threshold:", round(summary["threshold"], 4))
''')

M("## 5 · Leaderboard — every model, every metric")
C('''
import pandas as pd
from IPython.display import display, Image, Markdown
from pathlib import Path
R = Path(PROJECT_DIR) / "reports"; F = R / "figures"
board = pd.read_csv(R / "leaderboard_full.csv")
cols = ["model","family","accuracy","balanced_accuracy","precision","recall","specificity","f1","f2","mcc","cohen_kappa",
        "pr_auc","roc_auc","log_loss","brier","ece","fpr","fnr","tn","fp","fn","tp","threshold","cv_pr_auc","cv_pr_auc_std",
        "latency_ms","size_mb","fit_seconds","f1_at_0.5","mcnemar_p_vs_champion","is_champion"]
pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
display(board[cols].round(4).style.background_gradient(subset=["f1","pr_auc","mcc","roc_auc"], cmap="Greens")
        .background_gradient(subset=["ece","brier","log_loss"], cmap="Reds").highlight_between(subset=["is_champion"], left=1, right=1, color="gold"))
''')
C('''
for name in ["leaderboard_bars","metric_heatmap","pr_all_models","roc_all_models","calibration_all_models","speed_vs_quality"]:
    print(name); display(Image(str(F / f"{name}.png"), width=900))
''')

M("## 6 · Champion deep-dive: CIs, threshold, calibration, confusion matrix, robustness")
C('''
import json
s = json.load(open(R / "summary.json"))
ci = s["bootstrap_ci_test"]; m = s["metrics_test"]
display(pd.DataFrame({k: {"value": m[k], "95% CI low": ci[k]["low"], "95% CI high": ci[k]["high"]} for k in ci}).T.round(4))
print("Calibration comparison (validation):"); display(pd.read_json(R / "calibration_comparison.json").round(4))
for name in ["confusion_champion","threshold_sweep_champion","score_distribution_champion","robustness","top_features_logreg"]:
    p = F / f"{name}.png"
    if p.exists(): print(name); display(Image(str(p), width=800))
print("Spam recall under obfuscation attacks:"); display(pd.read_csv(R / "robustness.csv").round(4))
print("Metrics by message length:"); display(pd.read_csv(R / "slice_metrics.csv"))
''')
C('''
display(Markdown("### ❌ False positives (legit messages flagged as spam) — the costly error"))
display(pd.read_csv(R / "errors_false_positives.csv").head(10))
display(Markdown("### ❌ False negatives (missed spam)"))
display(pd.read_csv(R / "errors_false_negatives.csv").head(10))
display(Markdown((R / "MODEL_CARD.md").read_text()))
''')

M("## 7 · Try it: predictions + word-level explanations")
C('''
from spamguard.predictor import Predictor
pred = Predictor.from_champion()
tests = ["URGENT! Your parcel is held. Pay the $2.99 customs fee at http://dhl-track.xyz/pay today",
         "Hey, are we still on for dinner at 8?",
         "WINNER!! You have been selected for a £900 prize. Call 09061701461 now"]
for t in tests:
    r = pred.predict([t], explain=True)[0]
    ex = r["explanation"]
    print(f"{r['label'].upper():5s} P(spam)={r['spam_probability']:.3f} | {t[:70]}")
    print("      pushes → spam:", [(x['token'], round(x['weight'], 2)) for x in ex['top_spam_tokens'][:4]],
          "| → ham:", [(x['token'], round(x['weight'], 2)) for x in ex['top_ham_tokens'][:3]])
''')

M("""
## 8 · Monitoring demo: data drift
Normal traffic should be `ok`; a brand-new scam campaign should trigger an `alert` (PSI + KS tests on message features, OOV rate and score
distribution). An **Evidently** HTML report is also written to `reports/drift_report.html`.
""")
C('''
from spamguard.monitoring import run_check
logp = Path(PROJECT_DIR) / "logs" / "predictions.jsonl"
test_df = pd.read_csv(Path(PROJECT_DIR) / "data/processed/test.csv")
if logp.exists(): logp.unlink()
pred.predict(test_df["text"].tolist()[:300]);  print("normal traffic →", run_check()["status"])
logp.unlink()
pred.predict([f"Your parcel {i} is held. Pay customs fee $2.99 at https://dhl-track{i}.xyz/pay today, bitcoin accepted URGENT" for i in range(300)])
rep = run_check(); print("new scam wave →", rep["status"], "| drifted:", rep["drifted_features"][:6])
logp.unlink()
''')

M("## 9 · Human feedback loop + champion/challenger retraining demo")
C('''
from spamguard.retrain import retrain
# NOTE: demo rows are tagged source='demo'. Delete Drive/.../data/feedback/feedback.jsonl before real use.
# simulate analysts correcting the model (in production this comes from the /feedback API or the Gradio buttons)
fn = pd.read_csv(R / "errors_false_negatives.csv")["text"].tolist()
extra_spam = [f"URGENT! Verify your account now at http://secure-login{i}.xyz or it will be suspended" for i in range(30)]
for t in fn + extra_spam:  pred.record_feedback(t, "spam", source="demo")
for t in ["ok see you soon", "can you send me the notes?", "running late, start without me"] * 8: pred.record_feedback(t + f" ({hash(t) % 97})", "ham", source="demo")
res = retrain(promote=False, force=True)       # promote=True would register the challenger if it passes the gate
print(json.dumps(res, indent=2))
''')

M("""
## 10 · Launch the app
* **Gradio UI** (public share link, works from your phone): classify + highlighted explanation, feedback buttons, CSV batch scoring, model dashboard, drift monitor.
* **FastAPI** service running in the background on port 8000 (`/docs` has the Swagger UI inside the project; here we call it with `requests`).
""")
C('''
import threading, time, requests, uvicorn
from spamguard.api import create_app
server = uvicorn.Server(uvicorn.Config(create_app(enable_ui=False), host="127.0.0.1", port=8000, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start(); time.sleep(4)
print(requests.get("http://127.0.0.1:8000/model-info").json())
print(requests.post("http://127.0.0.1:8000/predict", json={"text": "Free entry! Text WIN to 80086 to claim your prize", "explain": False}).json())
''')
C('''
from spamguard.app_gradio import build_ui
demo = build_ui(lambda: pred)
demo.launch(share=True, debug=False)      # <- click the public gradio.live link
''')

M("## 11 · (Optional) Run the automated test-suite & package results for download")
C('''
!cd {PROJECT_DIR} && pip install -q pytest httpx && python -m pytest -q -x 2>&1 | tail -5
''')
C('''
import shutil, json
champ = json.load(open(f"{PROJECT_DIR}/models/champion.json"))["version"]
out = f"{PROJECT_DIR}/exports"; os.makedirs(out, exist_ok=True)
tmp = "/content/_export"; shutil.rmtree(tmp, ignore_errors=True)
shutil.copytree(f"{PROJECT_DIR}/reports", f"{tmp}/reports")
shutil.copytree(f"{PROJECT_DIR}/models/versions/{champ}", f"{tmp}/models/versions/{champ}")
shutil.copy(f"{PROJECT_DIR}/models/champion.json", f"{tmp}/models/champion.json")
shutil.copy(f"{PROJECT_DIR}/models/registry.json", f"{tmp}/models/registry.json")
archive = shutil.make_archive(f"{out}/spamguard_results_{champ}", "zip", tmp)
print("📦 saved to Drive:", archive)
# from google.colab import files; files.download(archive)     # <- uncomment to download straight to your computer
''')

M("""
## What's saved in `My Drive/sms-spam-mlops/`
| Folder | Contents |
|---|---|
| `data/raw`, `data/processed` | Kaggle download (cached) · cleaned train/val/test CSVs + manifest (data fingerprint) |
| `checkpoints/` | one `.joblib/.json/.npz` per model → **resume after disconnect** |
| `models/` | versioned, checksummed model registry + `champion.json` (promote / rollback) |
| `reports/` | leaderboard (CSV+MD), metrics.json, model card, error analysis, drift reference, **all figures** |
| `data/feedback`, `logs/` | human feedback · privacy-safe prediction logs |
| `mlflow.db`, `exports/` | experiment tracking · zipped results for download |

**Next:** copy the project to GitHub (CI is pre-configured), `docker compose up --build` to serve it, `dvc init` for data/model versioning.
""")

nb.metadata["colab"] = {"provenance": [], "name": "SpamGuard_Colab.ipynb"}
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebooks/SpamGuard_Colab.ipynb")
print("notebook written:", len(nb.cells), "cells")
