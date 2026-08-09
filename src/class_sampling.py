"""
Deterministic stratified sampling, cohort manifests, and rank-based splits.

Sampling, nesting, manifest round-trip and split logic were tested before deployment.
Replaces stain_norm.stratified_sample.

The old version threaded ONE random.Random across all class directories, so the
draw for class k depended on how many draws classes 0..k-1 had consumed. That
made the sample non-nested in n_per_class, and unstable under exclude_classes
or any change to the set of class folders on disk.

## THE CENTRAL IDEA: RANK
Each class (tissue type) is permuted exactly once, using a generator seeded from
(seed, class_name). A tile's RANK is its position in that permutation. Rank is
written into the manifest, and everything downstream is a rank threshold:

    rank <  n_train                 -> train
    n_train <= rank < n_train+n_val -> val
    rank <  b                       -> label-efficiency budget b (within train)

Because one permutation underlies all of them, subsets nest for free: the
budget-10 set is inside the budget-50 set, and every budget is inside train.
No downstream function draws any randomness of its own, so none of these splits
can drift when you change n_per_class, exclude a class, or reorder rows.

THE MANIFEST IS THE ARTIFACT OF RECORD
Regenerating a draw from a seed depends on random.shuffle's internals staying
fixed across Python versions, which is not a documented guarantee. So the
manifest CSV -- not the seed -- is what the feature cache is keyed to.

"""

import csv
import hashlib
import random
from pathlib import Path


def _class_rng(seed, class_name):
    """
    Per-class generator, seeded from the (seed, class_name) pair.

    blake2b rather than hash(): PYTHONHASHSEED randomises str hashing per
    process, so hash() would make this non-reproducible across runs.
    """
    digest = hashlib.blake2b(f"{seed}:{class_name}".encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def sample_cohort(root_path, n_per_class=None, seed=42, exclude_classes=(),
                  pattern="*.tif", strict=False):
    """
    Builds a cohort of tiles, class-grouped, each carrying its permutation rank.


    Parameters
    ----------
    n_per_class : int or None
        Tiles per class. None takes every tile, which is what you want for
        CRC-VAL-HE-7K: it is the test cohort and it is small, so use all of it
        and rely on balanced metrics rather than on subsampling.
    exclude_classes : tuple of str
        Default () keeps all nine classes. Pass ("BACK",) when building a
        cohort to FIT a Macenko target on -- pure background yields degenerate
        stain vectors -- but never when building a classification cohort, where
        BACK is a label.
    strict : bool
        Raise if a class has fewer tiles than n_per_class. Default False takes
        what is there, which is a quiet way to end up with an unbalanced
        "balanced" cohort.

    Returns
    -------
    list of dict
        Keys: path (Path), label (str), rank (int).
    """
    class_dirs = sorted(d for d in Path(root_path).iterdir() if d.is_dir())
    records = []
    for class_dir in class_dirs:
        if class_dir.name in exclude_classes:
            continue
        files = sorted(class_dir.glob(pattern))
        if not files:
            continue
        if strict and n_per_class is not None and len(files) < n_per_class:
            raise ValueError(
                f"{class_dir.name}: {len(files)} tiles < n_per_class={n_per_class}"
            )
        # Permute the whole class, then take a prefix. This -- not
        # rng.sample(files, n) -- is what makes draws nested in n.
        order = list(files)
        _class_rng(seed, class_dir.name).shuffle(order)
        if n_per_class is not None:
            order = order[:n_per_class]
        for rank, p in enumerate(order):
            records.append({"path": p, "label": class_dir.name, "rank": rank})
    return records


def class_index(records):
    """
    Fixed label -> integer mapping, sorted alphabetically.

    Derive it from the manifest and store it with every feature cache.
    """
    return {lab: i for i, lab in enumerate(sorted({r["label"] for r in records}))}


def write_manifest(records, out_path, root_path, spec):
    """
    Writes the cohort to CSV and returns (out_path, sha256 of the CSV).

    Small, commits cleanly, and doubles as the checksum manifest that keeps the
    datasets themselves out of git. Sampling parameters go in a comment header
    so the file is self-describing.
    """
    root_path = Path(root_path)
    out_path = Path(out_path).with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        for key in sorted(spec):
            fh.write(f"# {key}: {spec[key]}\n")
        fh.write(f"# root: {root_path.as_posix()}\n")
        fh.write(f"# n_tiles: {len(records)}\n")
        writer = csv.writer(fh)
        writer.writerow(["row", "relpath", "label", "rank"])
        for i, r in enumerate(records):
            # POSIX relpaths so a manifest written on Windows reads on Linux.
            writer.writerow([i, Path(r["path"]).relative_to(root_path).as_posix(),
                             r["label"], r["rank"]])

    return out_path, hashlib.sha256(out_path.read_bytes()).hexdigest()


def read_manifest(manifest_path, root_path):
    """
    Returns (records, spec). Inverse of write_manifest.
    """
    manifest_path = Path(manifest_path)
    root_path = Path(root_path)
    spec, body = {}, []
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        for ln in fh:
            if ln.startswith("#"):
                key, _, val = ln[1:].partition(":")
                spec[key.strip()] = val.strip()
            else:
                body.append(ln)
    records = [{"path": root_path / row["relpath"], "label": row["label"],
                "rank": int(row["rank"])} for row in csv.DictReader(body)]
    return records, spec


def verify_manifest(manifest_path, root_path, expected_sha256=None,
                    check_exists=True):
    """
    Integrity check to run at the top of any analysis notebook.

    Catches the silent case where the data directory was re-downloaded or
    re-extracted between the embedding pass and the analysis. Returns findings
    rather than raising, so a notebook can just print it.
    """
    manifest_path = Path(manifest_path)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    records, spec = read_manifest(manifest_path, root_path)
    missing = [r["path"] for r in records if not r["path"].exists()] if check_exists else []
    counts = {}
    for r in records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    return {
        "sha256": digest,
        "sha256_ok": None if expected_sha256 is None else digest == expected_sha256,
        "n_tiles": len(records),
        "per_class": dict(sorted(counts.items())),
        "n_missing": len(missing),
        "missing_examples": [p.as_posix() for p in missing[:5]],
        "spec": spec,
    }


def split_train_val(records, n_train_per_class, n_val_per_class=None):
    """
    Per-class train/val split by rank. Returns (train_rows, val_rows).
    n_val_per_class=None uses whatever remains after train.
    """
    by_class = {}
    for i, r in enumerate(records):
        by_class.setdefault(r["label"], []).append((r["rank"], i))

    train, val = [], []
    for lab in sorted(by_class):
        for rank, i in sorted(by_class[lab]):
            if rank < n_train_per_class:
                train.append(i)
            elif n_val_per_class is None or rank < n_train_per_class + n_val_per_class:
                val.append(i)
    return sorted(train), sorted(val)


def label_budgets(records, budgets, rows=None):
    """
    Picks which rows to train on at each point of the label-efficiency curve.
    """
    keep = set(range(len(records))) if rows is None else set(rows)
    out = {}
    for b in sorted(budgets):
        out[b] = sorted(i for i in keep if records[i]["rank"] < b)
    return out