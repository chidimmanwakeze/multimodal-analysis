"""
train_classifier.py

Phase 0 update: the original classifier (per-modality Linear projection ->
concatenate -> Linear classifier) has NO non-linear activation function
anywhere in it. Stacking two Linear layers with nothing non-linear between
them is mathematically still just one linear transformation -- it is not
more expressive than plain logistic regression, regardless of how many
"layers" it appears to have. This is almost certainly what "too simple"
meant in PI feedback.

This version adds a genuine MLP option (non-linear activation after each
modality's projection, plus a non-linear hidden layer before the final
classifier), and runs it SIDE BY SIDE with the original linear version on
the same data every time -- so any performance difference can be
attributed specifically to the architecture change, not to a different
dataset or evaluation procedure.

Usage (unchanged):
    python train_classifier.py --dataset results/multimodal_dataset.npz --cv 3
    python train_classifier.py --dataset results/clinical_pathology_dataset.npz --cv 3

New flags:
    --activation {relu, gelu, tanh}   (default relu)
    --hidden-dim N                     (default 128)
    --dropout P                        (default 0.2)
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


class LinearMultimodalClassifier(nn.Module):
    """The original architecture: per-modality Linear projection ->
    concatenate -> Linear classifier. NO non-linear activation anywhere --
    kept here specifically so it can be compared against the MLP version
    on identical data, to isolate the effect of adding non-linearity."""

    def __init__(self, input_dims: list, proj_dim: int = 128, n_classes: int = 6):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(d, proj_dim) for d in input_dims])
        self.classifier = nn.Linear(proj_dim * len(input_dims), n_classes)

    def forward(self, xs: list):
        projected = [proj(x) for proj, x in zip(self.projections, xs)]
        combined = torch.cat(projected, dim=1)
        return self.classifier(combined)


class MLPMultimodalClassifier(nn.Module):
    """Adds two things the linear version lacks: (1) a non-linear activation
    after each modality's projection, and (2) one or more non-linear hidden
    layers (with dropout, given small sample sizes) before the final
    classifier. hidden_dims is a LIST -- [128] gives one 128-unit hidden
    layer (the original default), [256, 64] gives two layers, [] gives
    zero (degenerates to a single-hidden-layer-free MLP, useful for
    isolating how much the hidden layer itself is contributing vs. just
    the per-modality activations)."""

    def __init__(self, input_dims: list, proj_dim: int = 128, hidden_dims: list = None,
                 n_classes: int = 6, activation: str = "relu", dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128]
        act_cls = ACTIVATIONS[activation]
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(d, proj_dim), act_cls()) for d in input_dims
        ])

        layers = []
        in_dim = proj_dim * len(input_dims)
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_cls())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.hidden = nn.Sequential(*layers)  # empty Sequential if hidden_dims=[] -- acts as identity
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, xs: list):
        projected = [proj(x) for proj, x in zip(self.projections, xs)]
        combined = torch.cat(projected, dim=1)
        hidden = self.hidden(combined)
        return self.classifier(hidden)


def train_pytorch_model(X_list_tr, y_tr, X_list_te, n_classes: int, architecture: str = "mlp",
                         epochs: int = 100, lr: float = 1e-3, device: str = "cpu",
                         class_weight: bool = True, activation: str = "relu",
                         proj_dim: int = 128, hidden_dims: list = None, dropout: float = 0.2):
    input_dims = [X.shape[1] for X in X_list_tr]
    if architecture == "linear":
        model = LinearMultimodalClassifier(input_dims, proj_dim=proj_dim, n_classes=n_classes).to(device)
    else:
        model = MLPMultimodalClassifier(input_dims, proj_dim=proj_dim, hidden_dims=hidden_dims,
                                         n_classes=n_classes, activation=activation,
                                         dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if class_weight:
        counts = np.bincount(y_tr, minlength=n_classes)
        weights = len(y_tr) / (n_classes * np.maximum(counts, 1))
        weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_t)
    else:
        criterion = nn.CrossEntropyLoss()

    X_tr_t = [torch.tensor(X, dtype=torch.float32, device=device) for X in X_list_tr]
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_tr_t)
        loss = criterion(logits, y_tr_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        X_te_t = [torch.tensor(X, dtype=torch.float32, device=device) for X in X_list_te]
        logits = model(X_te_t)
        preds = logits.argmax(dim=1).cpu().numpy()

    return preds, model


def evaluate(y_true, y_pred, label_names, name: str):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n=== {name} ===")
    print(f"  Accuracy:     {acc:.3f}")
    print(f"  F1 (macro):   {f1_macro:.3f}")
    print(f"  F1 (weighted):{f1_weighted:.3f}")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(y_true, y_pred)
    header = "         " + " ".join(f"{l[:8]:>8s}" for l in label_names)
    print(header)
    for label, row in zip(label_names, cm):
        print(f"{label[:8]:>8s} " + " ".join(f"{v:>8d}" for v in row))
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def get_modality_arrays(data):
    modalities = list(data["modalities"])
    X_list = [data[f"X_{m}"] for m in modalities]
    return modalities, X_list


def run_all_methods(X_list, y, idx_tr, idx_te, label_names, n_classes, modalities, args):
    """Runs majority baseline, logistic regression, LINEAR multimodal, and
    MLP multimodal on the same train/test split -- so all four are directly
    comparable. Returns the metrics dict for each, keyed by method name."""
    results = {}

    majority_class = np.bincount(y[idx_tr]).argmax()
    majority_preds = np.full(len(idx_te), majority_class)
    results["majority"] = evaluate(y[idx_te], majority_preds, label_names, "Baseline: majority class")

    X_all = np.hstack(X_list)
    lr_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr_model.fit(X_all[idx_tr], y[idx_tr])
    lr_preds = lr_model.predict(X_all[idx_te])
    results["logreg"] = evaluate(y[idx_te], lr_preds, label_names,
                                  "Baseline: logistic regression (raw concat)")

    X_list_tr = [X[idx_tr] for X in X_list]
    X_list_te = [X[idx_te] for X in X_list]

    linear_preds, _ = train_pytorch_model(X_list_tr, y[idx_tr], X_list_te, n_classes=n_classes,
                                           architecture="linear", proj_dim=args.proj_dim, epochs=args.epochs)
    results["linear"] = evaluate(y[idx_te], linear_preds, label_names,
                                  f"PyTorch multimodal -- LINEAR (no activation, original architecture, "
                                  f"{len(modalities)} modalities: {', '.join(modalities)})")

    mlp_preds, _ = train_pytorch_model(X_list_tr, y[idx_tr], X_list_te, n_classes=n_classes,
                                        architecture="mlp", activation=args.activation,
                                        proj_dim=args.proj_dim, hidden_dims=args.hidden_dims,
                                        dropout=args.dropout, epochs=args.epochs)
    results["mlp"] = evaluate(y[idx_te], mlp_preds, label_names,
                               f"PyTorch multimodal -- MLP ({args.activation} activation, "
                               f"proj_dim={args.proj_dim}, hidden_dims={args.hidden_dims}, "
                               f"{len(modalities)} modalities: {', '.join(modalities)})")

    return results


def single_split_run(data, test_size: float, seed: int, args):
    modalities, X_list = get_modality_arrays(data)
    y = data["y"]
    label_names = list(data["label_names"])
    n_classes = len(label_names)

    print(f"Modalities in this dataset: {modalities}")

    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    print(f"Train: {len(idx_tr)} patients, Test: {len(idx_te)} patients")

    run_all_methods(X_list, y, idx_tr, idx_te, label_names, n_classes, modalities, args)


def cv_run(data, n_splits: int, seed: int, args):
    modalities, X_list = get_modality_arrays(data)
    y = data["y"]
    label_names = list(data["label_names"])
    n_classes = len(label_names)

    print(f"Modalities in this dataset: {modalities}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_true = []
    all_preds = {"majority": [], "logreg": [], "linear": [], "mlp": []}

    X_all = np.hstack(X_list)

    for fold, (idx_tr, idx_te) in enumerate(skf.split(X_list[0], y)):
        X_list_tr = [X[idx_tr] for X in X_list]
        X_list_te = [X[idx_te] for X in X_list]

        majority_class = np.bincount(y[idx_tr]).argmax()
        all_preds["majority"].extend(np.full(len(idx_te), majority_class))

        lr_model = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr_model.fit(X_all[idx_tr], y[idx_tr])
        all_preds["logreg"].extend(lr_model.predict(X_all[idx_te]))

        linear_preds, _ = train_pytorch_model(X_list_tr, y[idx_tr], X_list_te, n_classes=n_classes,
                                               architecture="linear", proj_dim=args.proj_dim, epochs=args.epochs)
        all_preds["linear"].extend(linear_preds)

        mlp_preds, _ = train_pytorch_model(X_list_tr, y[idx_tr], X_list_te, n_classes=n_classes,
                                            architecture="mlp", activation=args.activation,
                                            proj_dim=args.proj_dim, hidden_dims=args.hidden_dims,
                                            dropout=args.dropout, epochs=args.epochs)
        all_preds["mlp"].extend(mlp_preds)

        all_true.extend(y[idx_te])
        print(f"  Fold {fold + 1}/{n_splits}: linear acc={accuracy_score(y[idx_te], linear_preds):.3f}  "
              f"mlp acc={accuracy_score(y[idx_te], mlp_preds):.3f}  (test n={len(idx_te)})")

    print(f"\n--- Aggregated across all {n_splits} folds "
          f"(every patient appears in the test set exactly once) ---")
    evaluate(all_true, all_preds["majority"], label_names, "Baseline: majority class")
    evaluate(all_true, all_preds["logreg"], label_names, "Baseline: logistic regression (raw concat)")
    evaluate(all_true, all_preds["linear"], label_names,
             f"PyTorch multimodal -- LINEAR (no activation, original architecture, "
             f"{len(modalities)} modalities: {', '.join(modalities)})")
    evaluate(all_true, all_preds["mlp"], label_names,
             f"PyTorch multimodal -- MLP ({args.activation} activation, proj_dim={args.proj_dim}, "
             f"hidden_dims={args.hidden_dims}, {len(modalities)} modalities: {', '.join(modalities)})")


def single_modality_check(data, n_splits: int, seed: int):
    modalities, X_list = get_modality_arrays(data)
    y = data["y"]
    label_names = list(data["label_names"])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    print("\n=== Diagnostic: each modality alone (logistic regression) ===")
    for name, X in zip(modalities, X_list):
        all_true, all_pred = [], []
        for idx_tr, idx_te in skf.split(X, y):
            lr_model = LogisticRegression(max_iter=2000, class_weight="balanced")
            lr_model.fit(X[idx_tr], y[idx_tr])
            all_pred.extend(lr_model.predict(X[idx_te]))
            all_true.extend(y[idx_te])
        acc = accuracy_score(all_true, all_pred)
        f1m = f1_score(all_true, all_pred, average="macro", zero_division=0)
        print(f"  {name:<12s} alone: accuracy={acc:.3f}  F1(macro)={f1m:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="results/multimodal_dataset.npz")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=None,
                     help="if set, run stratified k-fold cross-validation with this many "
                          "folds instead of a single train/test split.")
    ap.add_argument("--activation", choices=list(ACTIVATIONS.keys()), default="relu",
                     help="non-linear activation used in the MLP architecture (default relu)")
    ap.add_argument("--proj-dim", type=int, default=128,
                     help="size each modality gets projected down to before concatenation "
                          "(default 128, same for linear and MLP)")
    ap.add_argument("--hidden-dims", type=str, default="128",
                     help="comma-separated hidden layer sizes for the MLP, e.g. '128' for one "
                          "128-unit layer (default, matches the original architecture), '64' for "
                          "a smaller single layer, '256,64' for two layers, or '' (empty string) "
                          "for zero hidden layers -- useful for isolating how much the hidden "
                          "layer itself is contributing vs. just the per-modality activations.")
    ap.add_argument("--dropout", type=float, default=0.2,
                     help="dropout probability in each MLP hidden layer (default 0.2)")
    ap.add_argument("--epochs", type=int, default=100,
                     help="training epochs for BOTH linear and MLP models (default 100). "
                          "If MLP consistently underperforms linear regardless of size, try "
                          "increasing this -- non-linear models are typically harder to "
                          "optimize than linear ones and may need more training time to "
                          "reach comparable performance, not necessarily a capacity problem.")
    args = ap.parse_args()
    args.hidden_dims = [int(x) for x in args.hidden_dims.split(",") if x.strip()] if args.hidden_dims else []

    data = np.load(args.dataset, allow_pickle=True)
    n_patients = len(data["y"])
    n_classes = len(data["label_names"])
    print(f"Dataset: {n_patients} patients, {n_classes} classes")
    if n_patients < 200:
        print(f"NOTE: {n_patients} patients across {n_classes} classes is a small-sample "
              f"classification problem. Treat any single accuracy number with caution -- "
              f"prefer --cv for a more stable estimate.\n")

    if args.cv:
        min_class_count = np.bincount(data["y"]).min()
        if min_class_count < args.cv:
            print(f"NOTE: your smallest class has only {min_class_count} patient(s), which is "
                  f"fewer than --cv {args.cv}. Stratified k-fold can't guarantee an even split "
                  f"in that case. Consider re-running with --cv {min_class_count} instead.\n")
        cv_run(data, args.cv, args.seed, args)
        single_modality_check(data, args.cv, args.seed)
    else:
        single_split_run(data, args.test_size, args.seed, args)


if __name__ == "__main__":
    main()
