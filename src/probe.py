"""
Linear probe on cached frozen features, with seed-averaged bootstrap CIs.

The split is NOT random: train/val come from manifest rank thresholds
(class_sampling.split_train_val), so the data is identical for every seed.
Seeds therefore vary only what the probe itself draws -- weight init and
minibatch order -- and the seed spread reported here is optimiser variance, not
sampling variance. Sampling uncertainty on the evaluation tiles is what the
bootstrap measures.
"""

import numpy as np

from . import class_sampling as cs


def make_split(data, n_train=1000, n_val=250):
    """(train_rows, val_rows) from a feature cache's labels and ranks."""
    records = [{"label": str(l), "rank": int(r)}
               for l, r in zip(data["labels"], data["ranks"])]
    return cs.split_train_val(records, n_train, n_val)


def encode_labels(labels, classes):
    lookup = {c: i for i, c in enumerate(classes)}
    return np.array([lookup[str(l)] for l in labels], dtype=np.int64)


def fit_probe(x_train, y_train, n_classes, seed, wd=1e-3, epochs=100, lr=1e-3,
              batch_size=256, device=None):
    """
    Multinomial logistic regression (a single linear layer, softmax
    cross-entropy, L2 penalty `wd` on the weights). Features are standardised
    with TRAIN statistics, which are returned so val/test can be transformed
    identically. Fixed epoch count, no early stopping: nothing is tuned on val
    inside the fit. Returns a predict(x) -> int array closure.
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mu = x_train.mean(0, keepdims=True)
    sd = x_train.std(0, keepdims=True) + 1e-6

    gen = torch.Generator().manual_seed(seed)
    x = torch.as_tensor((x_train - mu) / sd, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.long, device=device)
    w = (torch.randn(x.shape[1], n_classes, generator=gen) * 0.01).to(device).requires_grad_()
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)

    for _ in range(epochs):
        order = torch.randperm(len(x), generator=gen).to(device)
        for i in range(0, len(x), batch_size):
            idx = order[i:i + batch_size]
            loss = torch.nn.functional.cross_entropy(x[idx] @ w + b, y[idx]) \
                + 0.5 * wd * (w ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

    w_np, b_np = w.detach().cpu().numpy(), b.detach().cpu().numpy()

    def predict(x_new):
        return (((x_new - mu) / sd) @ w_np + b_np).argmax(1)

    return predict


def confusion(y_true, y_pred, k):
    """k x k counts; rows = true class, columns = predicted."""
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def macro_f1(cm):
    """Macro-F1 from a confusion matrix; a class with no support or predictions scores 0."""
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)))


def bootstrap_macro_f1(y_true, preds, k, n_boot=1000, seed=0, alpha=0.05):
    """
    preds: (n_seeds, n) predictions. Each bootstrap resamples TILES once and
    scores every seed on that same resample, then averages over seeds, so the
    interval reflects evaluation-set sampling around the seed-mean score.
    Returns (point, lo, hi, seed_std): point is the seed-mean macro-F1 on the
    full set, seed_std the spread across seeds.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    per_seed = [macro_f1(confusion(y_true, p, k)) for p in preds]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = np.mean([macro_f1(confusion(y_true[idx], p[idx], k)) for p in preds])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(per_seed)), float(lo), float(hi), float(np.std(per_seed))


def run_probe(nct, crc, seeds=(0, 1, 2, 3, 4), wd=1e-3, n_train=1000, n_val=250):
    """
    nct, crc: dicts from feat_enc.load_features. Trains on NCT train rows,
    evaluates on NCT val rows (in-distribution) and on every CRC row.
    Returns {"classes", "y_val", "y_crc", "pred_val", "pred_crc"} with pred_*
    shaped (n_seeds, n).
    """
    classes = sorted({str(l) for l in nct["labels"]})
    if classes != sorted({str(l) for l in crc["labels"]}):
        raise ValueError("NCT and CRC cohorts have different class sets")
    k = len(classes)
    train, val = make_split(nct, n_train, n_val)
    y = encode_labels(nct["labels"], classes)
    y_crc = encode_labels(crc["labels"], classes)

    pred_val, pred_crc = [], []
    for s in seeds:
        predict = fit_probe(nct["features"][train], y[train], k, seed=s, wd=wd)
        pred_val.append(predict(nct["features"][val]))
        pred_crc.append(predict(crc["features"]))
    return {"classes": classes, "y_val": y[val], "y_crc": y_crc,
            "pred_val": np.stack(pred_val), "pred_crc": np.stack(pred_crc)}
