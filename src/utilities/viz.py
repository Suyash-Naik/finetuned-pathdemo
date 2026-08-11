from pathlib import Path
import matplotlib.pyplot as plt
import shutil
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from bioio import BioImage
import bioio_tifffile
import random

def plot_class_counts(counts, title="Patch counts per tissue class"):
    """Bar chart of class counts, with counts labeled above each bar."""
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.keys(), counts.values(), color="#80c7f7")
    ax.set_ylabel("Number of patches")
    ax.set_title(title)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:,}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_class_sample_grid(root_path, n_samples=4, reader=bioio_tifffile.reader.Reader, seed=42):
    """
    Displays a grid of sample images: one row per class, n_samples columns per row.
    Each panel is labeled with its class name for clarity.
    """
    class_dirs = sorted([d for d in Path(root_path).iterdir() if d.is_dir()])
    rng = random.Random(seed)

    n_classes = len(class_dirs)
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(n_samples * 2.2, n_classes * 2.2))
    fig.suptitle("Sample patches per tissue class, unnormalized", fontsize=18)
    for row, class_dir in enumerate(class_dirs):
        tif_files = list(class_dir.glob("*.tif"))
        if not tif_files:
            print(f"Warning: no .tif files found in {class_dir.name}")
            continue

        sample_files = rng.sample(tif_files, min(n_samples, len(tif_files)))

        for col in range(n_samples):
            ax = axes[row, col]
            if col < len(sample_files):
                img = BioImage(sample_files[col], reader=reader)
                image_data = img.get_image_data("YXS")
                ax.imshow(image_data)
            ax.axis("off")
            ax.set_title(class_dir.name, fontsize=16)
    
    plt.tight_layout()
    plt.show()

def load_class_samples(root_path, n_samples=4, reader=bioio_tifffile.reader.Reader, seed=42):
    """
    Loads n_samples random images per class subfolder.
    Returns dict: {class_name: [np.array, ...]}
    """
    class_dirs = sorted([d for d in Path(root_path).iterdir() if d.is_dir()])
    rng = random.Random(seed)
    samples = {}

    for class_dir in class_dirs:
        tif_files = list(class_dir.glob("*.tif"))
        if not tif_files:
            print(f"Warning: no .tif files found in {class_dir.name}")
            continue
        sample_files = rng.sample(tif_files, min(n_samples, len(tif_files)))
        samples[class_dir.name] = [
            BioImage(f, reader=reader).get_image_data("YXS") for f in sample_files
        ]
    return samples

def load_manifest_samples(cohorts, classes, n_samples=4,
                          reader=bioio_tifffile.reader.Reader, annotate=None):
    """
    Loads tiles from cohort manifests, grouped by (cohort, class).

    Parameters
    ----------
    cohorts : dict of {str: list of dict}
        Maps a display name to the records list returned by
        class_sampling.read_manifest. Insertion order is preserved in the
        dropdown.
    classes : sequence of str
        Class labels to include. One group per (cohort, class) pair.
    n_samples : int, optional
        Tiles per group, taken as the lowest ranks in that class.
    reader : bioio reader class, optional
        Reader passed to BioImage.
    annotate : callable, optional
        Called as annotate(image_array); the value is formatted into each
        tile's caption. Pass stain_norm.tissue_fraction to show the quantity
        the normalization guard thresholds on, which is what ties the image to
        the rejection table. Kept as an argument rather than an import so this
        module does not depend on stain_norm.
 
    Returns
    -------
    tuple of (dict, dict)
        (samples, captions), both keyed by "<cohort> - <class>". samples maps
        to a list of HxWx3 arrays; captions maps to a list of strings.
    """
    samples, captions = {}, {}
    for cohort_name, records in cohorts.items():
        for label in classes:
            recs = sorted((r for r in records if r["label"] == label),
                          key=lambda r: r["rank"])[:n_samples]
            if not recs:
                continue
            key = f"{cohort_name} - {label}"
            imgs, caps = [], []
            for r in recs:
                img = BioImage(Path(r["path"]), reader=reader).get_image_data("YXS")
                imgs.append(img)
                cap = f"rank {r['rank']}"
                if annotate is not None:
                    cap += f" | {annotate(img):.2f}"
                caps.append(cap)
            samples[key] = imgs
            captions[key] = caps
    if not samples:
        raise ValueError("No groups to plot; check cohort names and class labels")
    return samples, captions
 
 
def plot_interactive_class_grid(samples, n_samples=4, captions=None,
                                title_prefix="Sample patches"):
    """
    Interactive grid of sample tiles with a dropdown to switch group.
 
    Accepts the output of either load_class_samples or load_manifest_samples.
 
    Parameters
    ----------
    samples : dict of {str: list of ndarray}
        Group name to list of HxWx3 arrays.
    n_samples : int, optional
        Number of subplot columns. Groups with fewer tiles leave trailing
        columns empty.
    captions : dict of {str: list of str}, optional
        Per-tile captions, same keys as samples. Rendered as subplot titles and
        swapped with the dropdown selection.
    title_prefix : str, optional
        Leading text of the figure title.
 
    Returns
    -------
    plotly.graph_objects.Figure
    """
    group_names = list(samples.keys())
 
    # Blank placeholders reserve the annotation slots the dropdown rewrites;
    # make_subplots only creates annotations for titles passed here.
    fig = make_subplots(rows=1, cols=n_samples, horizontal_spacing=0.02,
                        subplot_titles=[" "] * n_samples)
    base_annotations = [a.to_plotly_json() for a in fig.layout.annotations]
 
    # Record which group each trace belongs to rather than deriving it from
    # trace position: the previous version assumed every group contributed
    # exactly n_samples traces, so a short group silently shifted the
    # visibility mask for every group after it.
    trace_group = []
    for group_idx, name in enumerate(group_names):
        for col, img in enumerate(samples[name][:n_samples]):
            fig.add_trace(go.Image(z=img, visible=(group_idx == 0)),
                          row=1, col=col + 1)
            trace_group.append(group_idx)
 
    for i in range(1, n_samples + 1):
        fig.update_xaxes(visible=False, row=1, col=i)
        fig.update_yaxes(visible=False, row=1, col=i, scaleanchor=f"x{i}")
 
    def _annotations_for(name):
        if captions is None:
            return base_annotations
        caps = captions.get(name, [])
        out = []
        for idx, ann in enumerate(base_annotations):
            ann = dict(ann)
            ann["text"] = caps[idx] if idx < len(caps) else " "
            ann["font"] = {"size": 11}
            out.append(ann)
        return out
 
    buttons = []
    for group_idx, name in enumerate(group_names):
        buttons.append(dict(
            label=name, method="update",
            args=[{"visible": [g == group_idx for g in trace_group]},
                  {"title": f"{title_prefix} - {name}",
                   "annotations": _annotations_for(name)}],
        ))
 
    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, direction="down",
                          x=1.15, xanchor="left", y=1, yanchor="top")],
        title=f"{title_prefix} - {group_names[0]}",
        annotations=_annotations_for(group_names[0]),
        width=n_samples * 200 + 240,
        height=300,
        margin=dict(t=70, r=240, l=20, b=20),
        autosize=False,
    )
    return fig
 
 
def summarize_tissue_fraction(samples, tissue_fraction_fn):
    """
    Per-group summary of the measure the normalization guard thresholds on.
 
    The quantitative companion to the grid: a handful of tiles show what a
    group looks like, this shows whether those tiles were typical. Operates on
    the arrays already loaded by load_manifest_samples rather than re-reading.
 
    Parameters
    ----------
    samples : dict of {str: list of ndarray}
        Output of load_manifest_samples.
    tissue_fraction_fn : callable
        stain_norm.tissue_fraction.
 
    Returns
    -------
    pandas.DataFrame
        Indexed by group, with n and min/median/max tissue fraction.
    """
    import pandas as pd
 
    rows = []
    for name, imgs in samples.items():
        fracs = np.array([tissue_fraction_fn(img) for img in imgs])
        rows.append({"group": name, "n": len(fracs), "min": fracs.min(),
                     "median": float(np.median(fracs)), "max": fracs.max()})
    return pd.DataFrame(rows).set_index("group").round(3)

def compute_cross_cohort_color_shift(stats_a, stats_b, exclude_keys=("overall",)):
    """
    Computes Euclidean distance between mean RGB vectors for each class,
    comparing two cohorts' channel stats (e.g. NCT vs CRC-VAL).
    """
    classes = [k for k in stats_a.keys() if k not in exclude_keys and k in stats_b]
    shifts = {}
    for cls in classes:
        mean_a = np.array(stats_a[cls]["mean"])
        mean_b = np.array(stats_b[cls]["mean"])
        shifts[cls] = float(np.linalg.norm(mean_a - mean_b))
    return shifts


def plot_color_shift(shifts, title="Mean color shift: NCT vs CRC-VAL"):
    labels = list(shifts.keys())
    values = list(shifts.values())

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color="#e34948",
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        textfont=dict(size=11, color="#0b0b0b"),
        cliponaxis=False,
    ))
    fig.update_layout(
        title=title,
        yaxis_title="Euclidean distance (RGB mean)",
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        font=dict(color="#0b0b0b"),
        yaxis=dict(gridcolor="#e1e0d9", zerolinecolor="#c3c2b7",
                    range=[0, max(values) * 1.15]),
        xaxis=dict(showgrid=False),
        width=800, height=400,
        margin=dict(t=60, b=40, l=60, r=20),
    )
    fig.show()
    return fig

