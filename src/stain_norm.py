import random
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
import torchstain
from bioio import BioImage
import bioio_tifffile
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

# internal function that rescales a HWC uint8 array to a CHW float tensor in 0-255 range,
# matching torchstain's expected input format
_to_tensor_255 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255)
])


def load_image_array(image_path, reader=bioio_tifffile.reader.Reader):
    """Loads a single image as an HxWxC uint8 numpy array."""
    img = BioImage(image_path, reader=reader)
    return img.get_image_data("YXS")


def fit_macenko_normalizer(reference_path, reader=bioio_tifffile.reader.Reader):
    """
    Fits a Macenko normalizer to a single reference image.
    Returns the fitted normalizer 
    reuse across every normalize_image() call
    rather than re-fitting per image.
    """
    reference_array = load_image_array(reference_path, reader=reader)
    reference_tensor = _to_tensor_255(reference_array)

    normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
    normalizer.fit(reference_tensor)
    return normalizer



# ---------------------------------------------------------------------------
# Tissue filtering
#
# torchstain's Macenko only uses pixels where EVERY channel satisfies
#     OD = -ln((I + 1) / Io) >= beta
# i.e. I <= Io * exp(-beta) - 1  (~205.6 for Io=240, beta=0.15).
# Note the NATURAL log: the classic Macenko paper uses log10, torchstain does
# not, so don't reuse a log10-derived threshold here.
#
# On near-white tiles (NCT class BACK, and much of ADI) that pixel set is tiny
# or degenerate. eigh() on its covariance usually does NOT raise -- it returns
# an arbitrary basis, so you get a plausible-looking but meaningless stain
# matrix. Exception handling cannot catch this; so its better to have to filter up front.
# ---------------------------------------------------------------------------

_IO_DEFAULT = 240
_BETA_DEFAULT = 0.15


def tissue_fraction(image_array, Io=_IO_DEFAULT, beta=_BETA_DEFAULT):
    """Fraction of pixels Macenko would actually use, matching torchstain exactly."""
    arr = image_array.astype(np.float32)
    od = -np.log((arr + 1.0) / Io)
    return float((od.min(axis=2) >= beta).mean())


def _fit_target_from_path(path, backend='torch', reader=bioio_tifffile.reader.Reader,
                          min_tissue_frac=0.25):
    """
    Fits Macenko on one tile and returns ((HERef 3x2, maxCRef 2), None), or
    (None, reason) on rejection. Never raises.
    """
    try:
        arr = load_image_array(path, reader=reader)
        frac = tissue_fraction(arr)
        if frac < min_tissue_frac:
            return None, f"low tissue ({frac:.2f} < {min_tissue_frac})"

        normalizer = torchstain.normalizers.MacenkoNormalizer(backend=backend)
        normalizer.fit(_to_tensor_255(arr))

        he = np.asarray(normalizer.HERef.detach().cpu()).reshape(3, 2)
        maxc = np.asarray(normalizer.maxCRef.detach().cpu()).reshape(2)

        if not (np.isfinite(he).all() and np.isfinite(maxc).all()):
            return None, "non-finite matrix"
        # Columns should be unit-norm OD direction vectors; a badly degenerate
        # fit shows up here before it silently poisons the median.
        norms = np.linalg.norm(he, axis=0)
        if np.any(norms < 0.5) or np.any(norms > 1.5):
            return None, f"degenerate column norms {norms.round(3).tolist()}"
        return (he, maxc), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _fit_targets(paths, backend='torch', reader=bioio_tifffile.reader.Reader,
                 min_tissue_frac=0.25, max_workers=8, verbose=True):
    """
    Fits Macenko over `paths` in parallel. Returns (kept_paths, HE stack (n,3,2),
    maxC stack (n,2), Counter of rejection reasons).

    torch's own intra-op threading fights a ThreadPoolExecutor, so threads are
    pinned to 1 op-thread each for the duration and restored afterwards.
    """
    from collections import Counter

    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    kept_paths, hes, maxcs, reasons = [], [], [], Counter()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_fit_target_from_path, p, backend, reader, min_tissue_frac): p
                for p in paths
            }
            for future in as_completed(futures):
                path = futures[future]
                value, reason = future.result()
                if value is None:
                    reasons[reason.split(" (")[0]] += 1
                    continue
                he, maxc = value
                kept_paths.append(path)
                hes.append(he)
                maxcs.append(maxc)
    finally:
        torch.set_num_threads(prev_threads)

    if verbose:
        print(f"Fitted {len(kept_paths)}/{len(paths)} tiles.")
        for reason, count in reasons.most_common():
            print(f"  rejected {count:5d}  {reason}")

    if not kept_paths:
        raise RuntimeError(
            f"No valid stain matrices from {len(paths)} tiles. "
            f"Rejection reasons: {reasons.most_common()}"
        )
    return kept_paths, np.stack(hes), np.stack(maxcs), reasons

def estimate_median_target(image_paths, backend='torch', reader=bioio_tifffile.reader.Reader,
                           min_tissue_frac=0.25, max_workers=8, verbose=True):
    """
    Builds a normalizer whose target is the ELEMENTWISE MEDIAN of HERef and
    maxCRef across image_paths -- a virtual reference rather than one tile.

    Preferred over select_median_reference: it uses both quantities that
    normalize() actually consumes, and one outlier tile cannot set the target.
    Column-order canonicalisation is already handled inside torchstain
    (__find_HE keys on the red channel), so columns are comparable across tiles.
    """
    _, hes, maxcs, _ = _fit_targets(image_paths, backend=backend, reader=reader,
                                    min_tissue_frac=min_tissue_frac,
                                    max_workers=max_workers, verbose=verbose)

    median_he = np.median(hes, axis=0)
    # Elementwise median breaks the unit-norm property of each column; restore it,
    # since HERef columns are direction vectors in OD space.
    median_he = median_he / np.linalg.norm(median_he, axis=0, keepdims=True)
    median_maxc = np.median(maxcs, axis=0)

    if verbose:
        spread = hes.std(axis=0)
        print(f"HERef median:\n{median_he.round(4)}")
        print(f"HERef elementwise std:\n{spread.round(4)}")
        print(f"maxCRef median: {median_maxc.round(4)}  (n={len(hes)})")

    normalizer = torchstain.normalizers.MacenkoNormalizer(backend=backend)
    normalizer.HERef = torch.tensor(median_he, dtype=torch.float32)
    normalizer.maxCRef = torch.tensor(median_maxc, dtype=torch.float32)
    return normalizer


def select_median_reference(image_paths, n_sample=300, backend='torch',
                            reader=bioio_tifffile.reader.Reader, seed=42, max_workers=8,
                            min_tissue_frac=0.25, use_maxc=True, verbose=True):
    """
    Returns the path of the tile whose Macenko target is closest to the median
    across a subsample. Use this when you want a real, citable reference image;
    use estimate_median_target when you just want the best target values.

    use_maxc: include maxCRef in the distance. HERef columns are unit-norm
    (entries ~0-1) while maxCRef is ~1-2, so maxCRef is standardised before
    concatenation to stop it dominating the norm.
    """
    rng = random.Random(seed)
    image_paths = list(image_paths)
    sampled_paths = rng.sample(image_paths, min(n_sample, len(image_paths)))

    kept_paths, hes, maxcs, _ = _fit_targets(
        sampled_paths, backend=backend, reader=reader,
        min_tissue_frac=min_tissue_frac, max_workers=max_workers, verbose=verbose)

    features = hes.reshape(len(hes), -1)
    if use_maxc and len(maxcs) > 1:
        scale = maxcs.std(axis=0)
        scale[scale == 0] = 1.0
        features = np.concatenate([features, maxcs / scale], axis=1)

    median_feature = np.median(features, axis=0)
    distances = np.linalg.norm(features - median_feature, axis=1)
    return kept_paths[int(np.argmin(distances))]


def save_normalizer_target(normalizer, path):
    """
    Persists the 8 numbers that fully define a fitted Macenko target.
    normalize() recomputes the source matrices per image, so HERef and maxCRef
    are the ENTIRE state -- save these and the reference image is disposable.
    """
    np.savez(
        Path(path).with_suffix(".npz"),
        HERef=np.asarray(normalizer.HERef.detach().cpu()),
        maxCRef=np.asarray(normalizer.maxCRef.detach().cpu()),
    )
    return Path(path).with_suffix(".npz")


def load_normalizer_target(path, backend='torch'):
    """Rebuilds a normalizer from a saved target. Use for CRC-VAL-HE-7K."""
    data = np.load(Path(path).with_suffix(".npz"))
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend=backend)
    normalizer.HERef = torch.tensor(data["HERef"], dtype=torch.float32)
    normalizer.maxCRef = torch.tensor(data["maxCRef"], dtype=torch.float32)
    return normalizer
def normalize_array(image_array, normalizer, min_tissue_frac=0.25):
    """
    Parameters
    ----------
    image_array : ndarray
        HxWx3 uint8 RGB tile.
    normalizer : torchstain.normalizers.MacenkoNormalizer
        Fitted normalizer supplying the target stain basis.
    min_tissue_frac : float, optional
        Minimum fraction of pixels exceeding the optical-density threshold
        (see tissue_fraction) required for normalization to be attempted.
 
    Returns
    -------
    tuple of (ndarray or None, str or None)
        On success, (HxWx3 uint8 array, None). On rejection, (None, reason).
        Reasons are short strings suitable for aggregating into per-class
        failure counts; "low tissue" and a raised exception support different
        conclusions about Macenko's applicability and are kept distinct.
    """
    # Guard, not try/except: on a near-white tile the decomposition usually
    # succeeds on an arbitrary basis rather than raising, so there is nothing
    # to catch. Eligibility has to be decided before it runs.
    frac = tissue_fraction(image_array)
    if frac < min_tissue_frac:
        return None, f"low tissue ({frac:.2f} < {min_tissue_frac})"
 
    try:
        norm, _, _ = normalizer.normalize(I=_to_tensor_255(image_array), stains=True)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
 
    arr = np.asarray(norm.detach().cpu() if hasattr(norm, "detach") else norm)
    if not np.isfinite(arr).all():
        return None, "non-finite output"
 
    # Reconstruction is Io * exp(-HERef @ C) and can exceed 255 slightly;
    # an unclipped uint8 cast wraps 260 to 4, scattering near-black pixels
    # through bright regions. Backend clipping is a version detail.
    return np.clip(arr, 0, 255).astype(np.uint8), None
 
 
def normalize_image(image_path, normalizer, reader=bioio_tifffile.reader.Reader,
                    min_tissue_frac=0.25, verbose=False):
    """
    Parameters
    ----------
    image_path : str or Path
        Path to a single tile.
    normalizer : torchstain.normalizers.MacenkoNormalizer
        Fitted normalizer supplying the target stain basis.
    reader : bioio reader class, optional
        Reader passed through to load_image_array.
    min_tissue_frac : float, optional
        Minimum fraction of pixels above the optical-density threshold.
    verbose : bool, optional
        Print the rejection reason for each rejected tile. Default False:
        rejection is expected at a non-trivial rate on background-heavy
        classes, and per-tile printing over a full cohort produces tens of
        thousands of lines.
 
    Returns
    -------
    ndarray or None
        HxWx3 uint8 normalized tile, or None if the tile was rejected.
    """
    image_array = load_image_array(image_path, reader=reader)
    arr, reason = normalize_array(image_array, normalizer,
                                  min_tissue_frac=min_tissue_frac)
    if arr is None and verbose:
        print(f"Normalization rejected for {image_path}: {reason}")
    return arr


def check_normalization(root_path, normalizer, n_samples=6,
                                reader=bioio_tifffile.reader.Reader, seed=69):
    """
    Picks n_samples random images from anywhere under root_path, normalizes each,
    and returns an interactive Plotly figure with raw on top, normalized below,
    for visual quality-checking before running normalization at scale.
    """
    all_files = list(Path(root_path).rglob("*.tif"))
    rng = random.Random(seed)
    sample_files = rng.sample(all_files, min(n_samples, len(all_files)))

    fig = make_subplots(rows=2, cols=n_samples, horizontal_spacing=0.02, vertical_spacing=0.08,
                         row_titles=["Raw", "Normalized"])

    for col, f in enumerate(sample_files):
        raw = load_image_array(f, reader=reader)
        norm = normalize_image(f, normalizer, reader=reader)

        fig.add_trace(go.Image(z=raw), row=1, col=col + 1)
        if norm is not None:
            fig.add_trace(go.Image(z=norm), row=2, col=col + 1)

        fig.update_xaxes(visible=False, row=1, col=col + 1)
        fig.update_xaxes(visible=False, row=2, col=col + 1, title_text=f.parent.name)
        fig.update_yaxes(visible=False, row=1, col=col + 1)
        fig.update_yaxes(visible=False, row=2, col=col + 1, scaleanchor=f"x{col+1}")

    fig.update_layout(
        title="Macenko normalization sanity check (raw vs. normalized)",
        width=n_samples * 180 + 60, height=420,
        margin=dict(t=80, l=60, r=20, b=20),
    )
    return fig

def collect_normalized_grid_data(root_path, normalizer, n_samples=3,
                                   reader=bioio_tifffile.reader.Reader, seed=42):
    """
    Normalizes sample images per class and returns the results as plain numpy
    arrays (or None for failures), with no plotting — separates the
    torch/normalization work from any rendering step.
    """
    class_dirs = sorted([d for d in Path(root_path).iterdir() if d.is_dir()])
    rng = random.Random(seed)
    grid_data = {}

    for class_dir in class_dirs:
        tif_files = list(class_dir.glob("*.tif"))
        sample_files = rng.sample(tif_files, min(n_samples, len(tif_files)))
        results = []
        for f in sample_files:
            norm = normalize_image(f, normalizer, reader=reader)
            results.append(norm)  # None on failure, array on success
        grid_data[class_dir.name] = results

    return grid_data

def plot_grid_from_data(grid_data, title="Sample patches per tissue class, Macenko-normalized"):
    """
    Plots pre-computed grid_data (from collect_normalized_grid_data) using Plotly
    instead of matplotlib, to avoid the torch/matplotlib native-library conflict.
    """
    class_names = list(grid_data.keys())
    n_classes = len(class_names)
    n_samples = max(len(v) for v in grid_data.values())

    fig = make_subplots(
        rows=n_classes, cols=n_samples,
        horizontal_spacing=0.01, vertical_spacing=0.02,
        row_titles=class_names,
    )

    for row, class_name in enumerate(class_names):
        for col, img in enumerate(grid_data[class_name]):
            r, c = row + 1, col + 1
            if img is not None:
                fig.add_trace(go.Image(z=img), row=r, col=c)
            else:
                fig.add_annotation(
                    text="N/A<br>(no stain signal)", showarrow=False,
                    xref=f"x{'' if r==1 and c==1 else r*n_samples - n_samples + c}",
                    yref=f"y{'' if r==1 and c==1 else r*n_samples - n_samples + c}",
                    x=0.5, y=0.5, font=dict(size=10),
                    row=r, col=c,
                )
            fig.update_xaxes(visible=False, row=r, col=c)
            fig.update_yaxes(visible=False, row=r, col=c, scaleanchor=f"x{c}" if r == 1 else None)

    fig.update_layout(
        title=title,
        width=n_samples * 160 + 100,
        height=n_classes * 130,
        margin=dict(t=80, l=100, r=20, b=20),
    )
    return fig


def _stats_from_accumulator(acc):
    n_px, total, total_sq = acc["n_px"], acc["sum"], acc["sum_sq"]
    mean = total / n_px
    var = np.maximum(total_sq / n_px - mean ** 2, 0.0)
    return mean, np.sqrt(var)


def compute_paired_channel_stats(root_path, normalizer, n_samples_per_class=200,
                                 reader=bioio_tifffile.reader.Reader, seed=42,
                                 verbose=True):
    """
    Returns (stats_raw, stats_norm) computed on the SAME tiles.

    Why paired: computing the two separately lets them diverge on which tiles
    they include. compute_normalized_channel_stats drops tiles where Macenko
    fails (BACK, thin ADI) while the raw pass keeps them, so the "variance
    reduction" would partly be measuring the removal of background tiles rather
    than any effect of normalization. Here a tile that fails to normalize is
    excluded from BOTH sides.

    Uses running sums rather than stacking pixels: 200 tiles x 9 classes is
    ~0.5 GB of uint8 held twice under the old concatenate approach.
    """
    class_dirs = sorted([d for d in Path(root_path).iterdir() if d.is_dir()])
    rng = random.Random(seed)
    stats_raw, stats_norm = {}, {}
    overall = {k: {"n_px": 0, "sum": np.zeros(3), "sum_sq": np.zeros(3)}
               for k in ("raw", "norm")}

    for class_dir in class_dirs:
        # sorted() matters: plain glob() order is filesystem-dependent, and on a
        # network share it is not guaranteed stable between calls.
        tif_files = sorted(class_dir.glob("*.tif"))
        if not tif_files:
            continue
        sample_files = rng.sample(tif_files, min(n_samples_per_class, len(tif_files)))

        acc = {k: {"n_px": 0, "sum": np.zeros(3), "sum_sq": np.zeros(3)}
               for k in ("raw", "norm")}
        n_ok = n_failed = 0

        for f in sample_files:
            raw = load_image_array(f, reader=reader)
            norm = normalize_image(f, normalizer, reader=reader)
            if norm is None:
                n_failed += 1
                continue
            n_ok += 1
            for key, img in (("raw", raw), ("norm", norm)):
                px = img.reshape(-1, 3).astype(np.float64)
                acc[key]["n_px"] += px.shape[0]
                acc[key]["sum"] += px.sum(axis=0)
                acc[key]["sum_sq"] += (px ** 2).sum(axis=0)
                overall[key]["n_px"] += px.shape[0]
                overall[key]["sum"] += px.sum(axis=0)
                overall[key]["sum_sq"] += (px ** 2).sum(axis=0)

        if n_ok == 0:
            if verbose:
                print(f"Warning: all {len(sample_files)} tiles failed in {class_dir.name}, skipping")
            continue

        for key, target in (("raw", stats_raw), ("norm", stats_norm)):
            mean, std = _stats_from_accumulator(acc[key])
            target[class_dir.name] = {"mean": mean.tolist(), "std": std.tolist(),
                                      "n": n_ok, "n_failed": n_failed}
        if verbose:
            print(f"{class_dir.name:6s} n={n_ok:4d} failed={n_failed:3d}")

    if not stats_raw:
        raise RuntimeError(f"No usable tiles under {root_path}")

    for key, target in (("raw", stats_raw), ("norm", stats_norm)):
        mean, std = _stats_from_accumulator(overall[key])
        target["overall"] = {"mean": mean.tolist(), "std": std.tolist(),
                             "n": sum(v["n"] for k, v in target.items() if k != "overall")}
    return stats_raw, stats_norm


def compute_channel_stats(root_path, n_samples_per_class=200,
                          reader=bioio_tifffile.reader.Reader, seed=42):
    """
    Per-class per-channel mean/std on RAW patches (no normalization).
    This is the `stats_before` that compute_variance_reduction expects; it was
    referenced in the module but never defined. Uses the same seed and sampling
    order as compute_normalized_channel_stats so the two are paired per tile.
    """
    class_dirs = sorted([d for d in Path(root_path).iterdir() if d.is_dir()])
    rng = random.Random(seed)
    stats = {}
    all_pixels = []

    for class_dir in class_dirs:
        tif_files = list(class_dir.glob("*.tif"))
        if not tif_files:
            continue
        sample_files = rng.sample(tif_files, min(n_samples_per_class, len(tif_files)))

        pixels = [load_image_array(f, reader=reader).reshape(-1, 3) for f in sample_files]
        pixels = np.concatenate(pixels, axis=0)
        stats[class_dir.name] = {
            "mean": pixels.mean(axis=0).tolist(),
            "std": pixels.std(axis=0).tolist(),
            "n": len(sample_files),
            "n_failed": 0,
        }
        all_pixels.append(pixels)

    if not all_pixels:
        raise RuntimeError(f"No class subdirectories with .tif files under {root_path}")

    overall = np.concatenate(all_pixels, axis=0)
    stats["overall"] = {"mean": overall.mean(axis=0).tolist(),
                        "std": overall.std(axis=0).tolist(),
                        "n": sum(v["n"] for v in stats.values())}
    return stats


def compute_normalized_channel_stats(root_path, normalizer, n_samples_per_class=200,
                                       reader=bioio_tifffile.reader.Reader, seed=42):
    """
    Same as compute_channel_stats, but normalizes each patch first.
    Failed normalizations (e.g. BACK patches with no stain signal) are skipped
    and counted separately rather than silently dropped.
    """
    class_dirs = sorted([d for d in Path(root_path).iterdir() if d.is_dir()])
    rng = random.Random(seed)
    stats = {}
    all_pixels = []

    for class_dir in class_dirs:
        tif_files = list(class_dir.glob("*.tif"))
        if not tif_files:
            continue
        sample_files = rng.sample(tif_files, min(n_samples_per_class, len(tif_files)))

        pixels = []
        n_failed = 0
        for f in sample_files:
            norm = normalize_image(f, normalizer, reader=reader)
            if norm is None:
                n_failed += 1
                continue
            pixels.append(norm.reshape(-1, norm.shape[-1]))

        if not pixels:
            print(f"Warning: all normalizations failed for {class_dir.name}, skipping")
            continue

        pixels = np.concatenate(pixels, axis=0)
        stats[class_dir.name] = {
            "mean": pixels.mean(axis=0).tolist(),
            "std": pixels.std(axis=0).tolist(),
            "n": len(sample_files) - n_failed,
            "n_failed": n_failed,
        }
        all_pixels.append(pixels)

    overall = np.concatenate(all_pixels, axis=0)
    stats["overall"] = {"mean": overall.mean(axis=0).tolist(), "std": overall.std(axis=0).tolist(),
                         "n": sum(v["n"] for k, v in stats.items())}
    return stats

def compute_variance_reduction(stats_before, stats_after, exclude_keys=("overall",)):
    """
    For each class, computes the % reduction in per-channel std after normalization.
    Positive = normalization tightened the distribution (expected direction).
    """
    classes = [k for k in stats_before.keys() if k not in exclude_keys and k in stats_after]
    reduction = {}
    for cls in classes:
        std_before = np.array(stats_before[cls]["std"])
        std_after = np.array(stats_after[cls]["std"])
        pct_reduction = ((std_before - std_after) / std_before) * 100
        reduction[cls] = pct_reduction.mean()  # averaged across R, G, B
    return reduction


def plot_variance_reduction(reduction, title="Std reduction after Macenko normalization"):
    fig = go.Figure()
    classes = list(reduction.keys())
    values = list(reduction.values())
    colors = ["#2ca02c" if v > 0 else "#c44e52" for v in values]
    fig.add_trace(go.Bar(x=classes, y=values, marker_color=colors))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title="Tissue class",
        yaxis_title="% reduction in channel std (mean of R,G,B)",
        width=800, height=400,
        margin=dict(t=80, l=60, r=20, b=20),
    )
    return fig

def plot_channel_stats_comparison(stats_before, stats_after, exclude_keys=("overall",)):
    """
    Combined figure: mean-color swatch strips (raw, normalized) stacked above
    a grouped bar chart comparing per-channel mean ± std, raw vs. normalized.
    """
    classes = [k for k in stats_before.keys() if k not in exclude_keys and k in stats_after]
    channels = ["R", "G", "B"]
    bar_colors = ["#d62728", "#2ca02c", "#1f77b4"]

    raw_swatch_colors = [
        f"rgb({stats_before[c]['mean'][0]:.0f},{stats_before[c]['mean'][1]:.0f},{stats_before[c]['mean'][2]:.0f})"
        for c in classes
    ]
    norm_swatch_colors = [
        f"rgb({stats_after[c]['mean'][0]:.0f},{stats_after[c]['mean'][1]:.0f},{stats_after[c]['mean'][2]:.0f})"
        for c in classes
    ]

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.1, 0.1, 0.8],
        vertical_spacing=0.04,
        subplot_titles=("Mean color — raw", "Mean color — normalized",
                         "Per-channel mean ± std: raw (faded) vs. normalized (solid)"),
    )

    fig.add_trace(go.Bar(x=classes, y=[1] * len(classes), marker_color=raw_swatch_colors,
                          hovertext=raw_swatch_colors, hoverinfo="text", showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=classes, y=[1] * len(classes), marker_color=norm_swatch_colors,
                          hovertext=norm_swatch_colors, hoverinfo="text", showlegend=False),
                  row=2, col=1)

    for c_idx, channel in enumerate(channels):
        means = [stats_before[cls]["mean"][c_idx] for cls in classes]
        stds = [stats_before[cls]["std"][c_idx] for cls in classes]
        fig.add_trace(go.Bar(x=classes, y=means, error_y=dict(type="data", array=stds),
                              name=f"{channel} (raw)", marker_color=bar_colors[c_idx], opacity=0.45,
                              legendgroup="raw", offsetgroup=c_idx),
                      row=3, col=1)

    for c_idx, channel in enumerate(channels):
        means = [stats_after[cls]["mean"][c_idx] for cls in classes]
        stds = [stats_after[cls]["std"][c_idx] for cls in classes]
        fig.add_trace(go.Bar(x=classes, y=means, error_y=dict(type="data", array=stds),
                              name=f"{channel} (normalized)", marker_color=bar_colors[c_idx], opacity=1.0,
                              legendgroup="normalized", offsetgroup=c_idx + 3),
                      row=3, col=1)

    fig.update_yaxes(visible=False, range=[0, 1], row=1, col=1)
    fig.update_yaxes(visible=False, range=[0, 1], row=2, col=1)
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_xaxes(visible=False, row=2, col=1)
    fig.update_yaxes(title_text="Pixel intensity (0-255)", row=3, col=1)

    fig.update_layout(
        barmode="group",
        title="Mean patch color and per-channel stats: raw vs. Macenko-normalized",
        legend=dict(title="Channel / condition"),
        width=1000, height=780,
    )
    return fig