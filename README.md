Anomaly Detection in Financial Transactions

This project explores how to detect high-risk transactions in a highly imbalanced financial dataset. The goal was not just overall accuracy, but correctly identifying rare high-risk cases, which are often missed by standard models.

Problem

In fraud or risk detection, models often achieve high accuracy by predicting the majority class, while failing to identify the rare but critical high-risk cases. This project focuses on improving detection of those high-risk transactions.

Approach

I tested several modeling strategies:

Random Forest (baseline)

Random Forest with SMOTE (class balancing)

XGBoost with class weighting

Ranger (weighted Random Forest)

Isolation Forest (unsupervised anomaly detection)

Hybrid model combining anomaly scores + multinomial logistic regression

Key Insight

Most models achieved high accuracy (~85%) but failed to detect high-risk transactions. This revealed a classic class imbalance problem, where models optimize for majority classes.

The hybrid approach performed best by:

using Isolation Forest to generate anomaly scores

incorporating those scores into a supervised model

tuning decision thresholds to prioritize high-risk detection

Results

High-risk recall improved to ~99%

High-risk F1 improved significantly (~0.40 vs ~0.12 baseline)

Tradeoff: increased false positives (precision ~25%)

This reflects a realistic product tradeoff: prioritizing catching risky events over minimizing alerts.

Takeaways

Accuracy is misleading in imbalanced problems

Threshold tuning can be as important as model choice

Hybrid approaches (unsupervised + supervised) can outperform standard models
