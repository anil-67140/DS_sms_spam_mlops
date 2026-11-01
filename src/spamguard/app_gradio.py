"""Gradio front-end: classify + explain, batch CSV, feedback, model dashboard, drift monitor.

    python -m spamguard.app_gradio            # local
    python -m spamguard.app_gradio --share    # public link (handy in Colab)
"""
from __future__ import annotations

import argparse
from typing import Callable

import gradio as gr
import pandas as pd

from .config import get_paths
from .predictor import Predictor

EXAMPLES = ["WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 now",
            "Hey, are we still meeting for lunch at 1pm tomorrow?",
            "URGENT! Your bank account is locked. Verify now at http://secure-bank-login.xyz to avoid suspension",
            "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121 to receive entry",
            "Can you pick up milk on your way home? Thanks!"]


def build_ui(get_predictor: Callable[[], Predictor | None]) -> gr.Blocks:
    def need():
        p = get_predictor()
        if p is None:
            raise gr.Error("No model loaded. Train one first: python -m spamguard.train")
        return p

    def classify(text, explain):
        if not text or not text.strip():
            return {}, [], "", "", ""
        p = need(); r = p.predict([text], explain=explain)[0]
        prob = r["spam_probability"]
        verdict = (f"## {'🚫 SPAM' if r['label'] == 'spam' else '✅ HAM (legitimate)'}\n"
                   f"P(spam) = **{prob:.3f}** · threshold = {r['threshold']:.3f} · model `{r['model_version']}` · {r['latency_ms']:.1f} ms")
        hl = []
        if explain and "explanation" in r:
            hl = [(t["token"] + " ", t["weight"]) for t in r["explanation"]["tokens"]]
        return {"spam": prob, "ham": 1 - prob}, hl, verdict, r["prediction_id"], str(prob)

    def send_feedback(text, label, pid, score):
        if not text or not pid:
            return "Classify a message first."
        need().record_feedback(text, label, prediction_id=pid, score=float(score), source="gradio")
        return f"✅ Thanks - recorded as **{label}**. This will be used in the next retraining cycle."

    def batch(file):
        if file is None:
            return None, None
        df = pd.read_csv(file if isinstance(file, str) else file.name)
        col = next((c for c in df.columns if c.lower() in ("text", "message", "sms", "v2")), df.columns[0])
        res = need().predict(df[col].astype(str).tolist())
        out = df.assign(spam_probability=[r["spam_probability"] for r in res], label=[r["label"] for r in res])
        path = str(get_paths().reports / "batch_predictions.csv"); out.to_csv(path, index=False)
        return out.head(200), path

    def dashboard():
        p = need(); md = p.bundle.metadata; m = md["metrics_test"]; ci = md.get("bootstrap_ci_test", {})
        rows = [{"metric": k, "value": round(m[k], 4),
                 "95% CI": f"{ci[k]['low']:.3f}-{ci[k]['high']:.3f}" if k in ci else ""}
                for k in ("accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "mcc", "fpr", "brier", "ece")]
        board = get_paths().reports / "leaderboard_full.csv"
        lb = pd.read_csv(board)[["model", "accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "mcc", "ece", "latency_ms"]].round(4) \
            if board.exists() else pd.DataFrame()
        figs = [str(f) for n in ("confusion_champion", "pr_all_models", "roc_all_models", "calibration_all_models",
                                 "threshold_sweep_champion", "leaderboard_bars", "robustness")
                if (f := get_paths().figures / f"{n}.png").exists()]
        info = f"**Champion:** `{md['model_name']}` · calibration `{md['calibration']}` · threshold `{p.threshold:.4f}` · version `{p.version}`"
        return info, pd.DataFrame(rows), lb, figs

    def drift():
        from .monitoring import drift_report, load_prediction_log, load_reference
        try:
            rep = drift_report(load_reference(), load_prediction_log(5000))
        except FileNotFoundError:
            return "Reference data missing - train a model first.", pd.DataFrame()
        if rep["status"] == "insufficient_data":
            return f"Need ≥ {rep['min_samples']} logged predictions (have {rep['n_current']}).", pd.DataFrame()
        rows = [{"feature": k, **{x: v[x] for x in ("psi", "ks_stat", "ref_mean", "cur_mean", "status")}}
                for k, v in rep["features"].items()]
        return f"### Drift status: **{rep['status'].upper()}** · drifted: {rep['drifted_features'] or 'none'}", pd.DataFrame(rows)

    with gr.Blocks(title="SpamGuard - SMS Spam Detection") as demo:
        gr.Markdown("# 🛡️ SpamGuard — SMS spam detection\nCalibrated probabilities, policy-based threshold, token-level explanations, human feedback loop.")
        with gr.Tab("Classify"):
            with gr.Row():
                with gr.Column(scale=3):
                    txt = gr.Textbox(label="SMS message", lines=4, placeholder="Paste a message…")
                    explain = gr.Checkbox(value=True, label="Explain (highlight words that pushed the score)")
                    btn = gr.Button("Classify", variant="primary")
                    gr.Examples(EXAMPLES, inputs=txt)
                with gr.Column(scale=2):
                    verdict = gr.Markdown(); lab = gr.Label(label="Probabilities", num_top_classes=2)
            hl = gr.HighlightedText(label="Explanation (red → spam signal, blue → ham signal)", combine_adjacent=False,
                                    show_legend=False)
            pid, score = gr.State(""), gr.State("")
            btn.click(classify, [txt, explain], [lab, hl, verdict, pid, score])
            txt.submit(classify, [txt, explain], [lab, hl, verdict, pid, score])
            gr.Markdown("**Was this wrong?** Your correction feeds the retraining pipeline.")
            with gr.Row():
                b1 = gr.Button("It is SPAM"); b2 = gr.Button("It is NOT spam"); fb = gr.Markdown()
            b1.click(lambda t, i, s: send_feedback(t, "spam", i, s), [txt, pid, score], fb)
            b2.click(lambda t, i, s: send_feedback(t, "ham", i, s), [txt, pid, score], fb)
        with gr.Tab("Batch (CSV)"):
            f = gr.File(label="CSV with a `text` column", file_types=[".csv"])
            go = gr.Button("Score file"); tbl = gr.Dataframe(label="Preview (first 200 rows)"); dl = gr.File(label="Download results")
            go.click(batch, f, [tbl, dl])
        with gr.Tab("Model dashboard"):
            refresh = gr.Button("Load / refresh"); info = gr.Markdown(); m_tbl = gr.Dataframe(label="Test metrics (95% bootstrap CI)")
            lb = gr.Dataframe(label="All models (test set)"); gal = gr.Gallery(label="Plots", columns=3, height="auto")
            refresh.click(dashboard, None, [info, m_tbl, lb, gal])
        with gr.Tab("Drift monitor"):
            d = gr.Button("Check drift on recent traffic"); dm = gr.Markdown(); dt = gr.Dataframe()
            d.click(drift, None, [dm, dt])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--share", action="store_true"); ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    pred = Predictor.from_champion()
    build_ui(lambda: pred).launch(share=a.share, server_name="0.0.0.0", server_port=a.port)
