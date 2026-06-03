"""Train logistic regression on hand-crafted features.

Usage:
    python -m src.features              # extract features -> cache/features.npz
    python -m src.train_lr              # train + eval
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report, confusion_matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="cache/features.npz")
    p.add_argument("--C", type=float, default=1.0, help="inverse regularization strength")
    args = p.parse_args()

    data = np.load(args.features, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    print(f"train: {len(X_train)} ({y_train.sum():.0f} fake)  val: {len(X_val)} ({y_val.sum():.0f} fake)")
    print(f"features: {X_train.shape[1]}")

    # check for NaN/inf
    bad = np.isnan(X_train).any(axis=1) | np.isinf(X_train).any(axis=1)
    if bad.any():
        print(f"dropping {bad.sum()} NaN/inf train rows")
        X_train, y_train = X_train[~bad], y_train[~bad]
    bad = np.isnan(X_val).any(axis=1) | np.isinf(X_val).any(axis=1)
    if bad.any():
        print(f"dropping {bad.sum()} NaN/inf val rows")
        X_val, y_val = X_val[~bad], y_val[~bad]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", class_weight="balanced")
        clf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train))
        val_pred = clf.predict(X_val)
        val_prob = clf.predict_proba(X_val)[:, 1]
        val_acc = accuracy_score(y_val, val_pred)
        val_bal = balanced_accuracy_score(y_val, val_pred)
        val_auc = roc_auc_score(y_val, val_prob)
        print(f"\nC={C}: train_acc={train_acc:.3f} val_acc={val_acc:.3f} val_bal={val_bal:.3f} val_auc={val_auc:.3f}")

    # detailed report for best C
    print("\n--- detailed (C=1.0) ---")
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced")
    clf.fit(X_train, y_train)
    val_pred = clf.predict(X_val)
    print(classification_report(y_val, val_pred, target_names=["real", "fake"]))

    cm = confusion_matrix(y_val, val_pred)
    print("confusion matrix (rows=actual, cols=predicted):")
    print(f"           pred_real  pred_fake")
    print(f"  real     {cm[0,0]:>9}  {cm[0,1]:>9}")
    print(f"  fake     {cm[1,0]:>9}  {cm[1,1]:>9}")

    # feature importance — names come from the npz when present (image features),
    # else fall back to the 25-feature video layout.
    coefs = np.abs(clf.coef_[0])
    if "feature_names" in data:
        names = [str(n) for n in data["feature_names"]]
    else:
        names = [
            "R_mean", "R_std", "G_mean", "G_std", "B_mean", "B_std",
            "tdiff_mean", "tdiff_std", "tdiff_px_mean", "tdiff_px_std",
            "dct_hi_R", "dct_hi_G", "dct_hi_B",
            "pvar_mean_R", "pvar_std_R", "pvar_mean_G", "pvar_std_G", "pvar_mean_B", "pvar_std_B",
            "noise_R", "noise_G", "noise_B",
            "edge_R", "edge_G", "edge_B",
        ]
    order = np.argsort(coefs)[::-1]
    print("top features:")
    for i in order[:10]:
        print(f"  {names[i]:15s} coef={clf.coef_[0][i]:+.4f} |coef|={coefs[i]:.4f}")


if __name__ == "__main__":
    main()
