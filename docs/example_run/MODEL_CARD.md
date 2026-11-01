# Model card - SpamGuard `20261031-153609-096c1a`

**Task:** binary classification of SMS messages (spam vs ham). **Champion:** `linear_svm`
(calibration: `none`), 17 models compared.

## Data
Source `local`; 5157 messages after cleaning (415 duplicates and
0 label-conflicts removed); spam rate 12.4%.
Stratified split: {"train": {"rows": 3609, "spam_rate": 0.1247}, "val": {"rows": 774, "spam_rate": 0.124}, "test": {"rows": 774, "spam_rate": 0.124}}.

## Decision policy
Threshold **0.7776** chosen (5-fold OOF on train+val (n=4383)) with strategy `min_precision`
(max recall subject to precision >= 0.99). Blocking a legitimate message is treated as costlier than missing spam.

## Test performance (held out, used once)
| metric | value |
|---|---|
| Precision | 1.0000 (95% CI 1.0000-1.0000) |
| Recall | 0.8854 (95% CI 0.8227-0.9479) |
| F1 | 0.9392 (95% CI 0.9027-0.9733) |
| MCC | 0.9334 (95% CI 0.8958-0.9700) |
| PR-AUC | 0.9911 (95% CI 0.9749-0.9995) |
| ROC-AUC | 0.9974 (95% CI 0.9919-0.9999) |
| False-positive rate | 0.0000 (95% CI 0.0000-0.0000) |
| Brier / ECE | 0.0070 / 0.0118 |
| Confusion (TN/FP/FN/TP) | 678/0/11/85 |

### Top 5 models
| model       |   precision |   recall |     f1 |   pr_auc |   roc_auc |    mcc |
|:------------|------------:|---------:|-------:|---------:|----------:|-------:|
| lightgbm    |           1 |   0.8229 | 0.9029 |   0.9931 |    0.9987 | 0.896  |
| extra_trees |           1 |   0.8125 | 0.8966 |   0.992  |    0.9988 | 0.8897 |
| rbf_svm     |           1 |   0.8333 | 0.9091 |   0.9917 |    0.9972 | 0.9023 |
| linear_svm  |           1 |   0.7812 | 0.8772 |   0.9911 |    0.9974 | 0.8705 |
| voting_soft |           1 |   0.7917 | 0.8837 |   0.9904 |    0.9973 | 0.8769 |

### Robustness (spam recall under evasion attacks)
| attack            |   spam_recall |   mean_spam_score |
|:------------------|--------------:|------------------:|
| clean (baseline)  |         0.885 |             0.917 |
| leetspeak         |         0.938 |             0.962 |
| letter spacing    |         0.885 |             0.915 |
| zero-width chars  |         0.854 |             0.903 |
| random caps       |         0.885 |             0.918 |
| punctuation noise |         0.854 |             0.899 |

## Limitations & ethical notes
* Trained on English SMS (2000s UK/Singapore-era corpus): new scams (crypto, delivery, bank-OTP phishing) and other
  languages are likely under-represented -> monitor drift and collect feedback.
* The test set is small; differences between the top models are usually inside the confidence interval.
* Messages are personal data: prediction logs store hashes + numeric features, never raw text by default.
* A spam flag should trigger *soft* actions (move to junk folder) rather than silent deletion.
