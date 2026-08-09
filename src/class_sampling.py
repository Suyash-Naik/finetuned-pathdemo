"""
Deterministic, nested stratified sampling + sample manifests.

PARTIALLY TESTED: nesting, exclusion-stability, manifest round-trip and
verify/hash were exercised against synthetic empty .tif files in a temp dir.
Never run against real NCT-CRC data, bioio, or Windows paths.

Replaces stain_norm.stratified_sample. The old version threaded ONE
random.Random across all class directories, so the draw for class k depended on
how many draws classes 0..k-1 had consumed. That made the sample non-nested in
n_per_class (breaking the label-efficiency curve, whose budget points must be
prefixes of one another) and unstable under exclude_classes or any change to
the set of class folders on disk.

Here each class gets its own generator, seeded from (seed, class_name), and the
sample is a prefix of one full permutation. Properties that follow:

    sample(n=500) is a strict subset of sample(n=1000), per class
    excluding a class does not perturb any other class
    adding/removing a class folder does not perturb any other class
    the per-class draw is independent of class iteration order

Regeneration from a seed is still an implementation detail of random.shuffle,
so the manifest written by write_manifest() -- not the seed -- is the artifact
of record. Regeneration is a consistency check against it.
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


def stratified_sample(root_path, n_per_class=40, seed=42, exclude_classes=("BACK",),
                      pattern="*.tif", shuffle=True, strict=False):
    """
    Samples n_per_class tiles from each class directory under root_path.

    NCT-CRC-HE-100K is not class-balanced, so a flat random sample over all
    100k tiles is dominated by the largest classes.

    exclude_classes defaults to ("BACK",) for the Macenko-target use case: pure
    background carries no stain information and produces degenerate fits. For
    building a CLASSIFICATION cohort pass exclude_classes=() -- BACK is one of
    the nine labels.

    Parameters
    ----------
    n_per_class : int
        Tiles per class. Samples for different n are nested: the n=500 draw is
        a subset of the n=1000 draw.
    shuffle : bool
        Shuffle the concatenated list before returning. Uses its own generator,
        so it does not disturb the per-class draws. Set False if you want the
        output grouped by class (easier to eyeball; irrelevant to the probe,
        which shuffles at split time anyway).
    strict : bool
        Raise if any class has fewer than n_per_class tiles. Default False
        silently takes what is there, which is how the old version behaved and
        is a quiet way to end up with an unbalanced "balanced" sample.

    Returns
    -------
    list of Path
    """
    class_dirs = sorted(d for d in Path(root_path).iterdir() if d.is_dir())
    sampled = []
    for class_dir in class_dirs:
        if class_dir.name in exclude_classes:
            continue
        files = sorted(class_dir.glob(pattern))
        if not files:
            continue
        if strict and len(files) < n_per_class:
            raise ValueError(
                f"{class_dir.name}: {len(files)} tiles < n_per_class={n_per_class}"
            )
        # Permute the whole class, then take a prefix. This -- not
        # rng.sample(files, n) -- is what makes the draw nested in n.
        order = list(files)
        _class_rng(seed, class_dir.name).shuffle(order)
        sampled.extend(order[:n_per_class])

    if shuffle:
        random.Random(seed).shuffle(sampled)
    return sampled


def write_manifest(paths, out_path, root_path, spec):
    """
    Writes the sample to CSV and returns (out_path, sha256 of the CSV).

    This file is the artifact of record for a cached feature set: the .npz
    stores features, this stores which tile each row came from. Small, commits
    cleanly, and doubles as the checksum manifest that keeps the datasets
    themselves out of git.

    spec : dict
        Sampling parameters (n_per_class, seed, exclude_classes, pattern) plus
        anything else needed to recreate the draw. Written as a comment header
        so the CSV stays a single self-describing file.
    """
    root_path = Path(root_path)
    out_path = Path(out_path).with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        for key in sorted(spec):
            fh.write(f"# {key}: {spec[key]}\n")
        fh.write(f"# root: {root_path.as_posix()}\n")
        fh.write(f"# n_tiles: {len(paths)}\n")
        writer = csv.writer(fh)
        writer.writerow(["index", "relpath", "label"])
        for i, p in enumerate(paths):
            p = Path(p)
            # POSIX relpaths so a manifest written on Windows reads on Linux.
            writer.writerow([i, p.relative_to(root_path).as_posix(), p.parent.name])

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return out_path, digest


def read_manifest(manifest_path, root_path):
    """
    Returns (paths, labels, spec). Inverse of write_manifest.

    Load the cohort from here rather than re-calling stratified_sample: it
    removes any dependence on random.shuffle's internals staying fixed across
    Python versions.
    """
    manifest_path = Path(manifest_path)
    root_path = Path(root_path)
    spec = {}
    rows = []
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        lines = [ln for ln in fh]
    body = []
    for ln in lines:
        if ln.startswith("#"):
            key, _, val = ln[1:].partition(":")
            spec[key.strip()] = val.strip()
        else:
            body.append(ln)
    for row in csv.DictReader(body):
        rows.append((root_path / row["relpath"], row["label"]))

    paths = [p for p, _ in rows]
    labels = [lab for _, lab in rows]
    return paths, labels, spec


def verify_manifest(manifest_path, root_path, expected_sha256=None,
                    check_exists=True):
    """
    Cheap integrity check to run at the top of any analysis notebook.

    Confirms the CSV hash still matches what the feature cache was built
    against, and (optionally) that every listed tile is still on disk. Catches
    the silent case where the data directory was re-downloaded or re-extracted
    between the embedding pass and the analysis.

    Returns a dict of findings rather than raising, so a notebook can print it.
    """
    manifest_path = Path(manifest_path)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    paths, labels, spec = read_manifest(manifest_path, root_path)

    missing = []
    if check_exists:
        missing = [p for p in paths if not p.exists()]

    return {
        "sha256": digest,
        "sha256_ok": None if expected_sha256 is None else digest == expected_sha256,
        "n_tiles": len(paths),
        "n_classes": len(set(labels)),
        "n_missing": len(missing),
        "missing_examples": [p.as_posix() for p in missing[:5]],
        "spec": spec,
    }


def nested_budgets(paths, labels, budgets):
    """
    Nested per-class label budgets for the label-efficiency curve.

    Given a cohort drawn by stratified_sample, returns {budget: indices} where
    the index set for a smaller budget is a subset of every larger one. The
    curve then varies label COUNT only; without nesting, each point is also a
    fresh random draw and the between-point scatter is partly sample luck
    dressed up as an effect.

    Call this on the TRAIN split only, after the train/val split, or the
    budgets will leak into validation.
    """
    by_class = {}
    for i, lab in enumerate(labels):
        by_class.setdefault(lab, []).append(i)
    # paths order is already a fixed permutation, so a prefix per class is
    # itself a valid nested draw -- no further randomisation needed.
    out = {}
    for b in sorted(budgets):
        idx = []
        for lab in sorted(by_class):
            idx.extend(by_class[lab][:b])
        out[b] = sorted(idx)
    return out