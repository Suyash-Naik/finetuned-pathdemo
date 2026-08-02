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


def plot_interactive_class_grid(samples, n_samples=4):
    class_names = list(samples.keys())

    fig = make_subplots(rows=1, cols=n_samples, horizontal_spacing=0.02)

    # One trace per (class, slot), placed in its dedicated subplot column
    for class_idx, class_name in enumerate(class_names):
        for col, img in enumerate(samples[class_name]):
            fig.add_trace(
                go.Image(z=img, visible=(class_idx == 0)),
                row=1, col=col + 1
            )

    # Hide axes on every subplot
    for i in range(1, n_samples + 1):
        fig.update_xaxes(visible=False, row=1, col=i)
        fig.update_yaxes(visible=False, row=1, col=i, scaleanchor=f"x{i}")

    buttons = []
    for class_idx, class_name in enumerate(class_names):
        visibility = [
            (i // n_samples) == class_idx
            for i in range(len(class_names) * n_samples)
        ]
        buttons.append(dict(
            label=class_name, method="update",
            args=[{"visible": visibility}, {"title": f"Sample patches — class: {class_name}"}]
        ))

    fig.update_layout(
        updatemenus=[dict(
            active=0, buttons=buttons, direction="down",
            x=1.15, xanchor="left", y=1, yanchor="top",
        )],
        title=f"Sample patches — class: {class_names[0]}",
        width=n_samples * 200 + 180,
        height=260,
        margin=dict(t=60, r=180, l=20, b=20),
        autosize=False,
    )
    return fig

def plot_channel_stats(stats, exclude_keys=("overall",)):
    """
    Visualizes per-class channel stats: a color-swatch strip showing each
    class's mean RGB color, plus a grouped bar chart of mean ± std per channel.
    """
    classes = [k for k in stats.keys() if k not in exclude_keys]
    channels = ["R", "G", "B"]

    fig, (ax_swatch, ax_bar) = plt.subplots(
        2, 1, figsize=(max(8, len(classes) * 1.1), 7),
        gridspec_kw={"height_ratios": [1, 3]}
    )

    # --- Swatch strip ---
    for i, cls in enumerate(classes):
        mean_rgb = np.array(stats[cls]["mean"]) / 255.0
        ax_swatch.add_patch(plt.Rectangle((i, 0), 1, 1, color=mean_rgb))
        ax_swatch.text(i + 0.5, -0.15, cls, ha="center", va="top", fontsize=10)
    ax_swatch.set_xlim(0, len(classes))
    ax_swatch.set_ylim(-0.3, 1)
    ax_swatch.axis("off")
    ax_swatch.set_title("Mean patch color per class (unnormalized)", fontsize=13)

    # --- Grouped bar chart with error bars ---
    x = np.arange(len(classes))
    width = 0.25
    colors = ["#d62728", "#2ca02c", "#1f77b4"]  # R, G, B

    for c_idx, channel in enumerate(channels):
        means = [stats[cls]["mean"][c_idx] for cls in classes]
        stds = [stats[cls]["std"][c_idx] for cls in classes]
        ax_bar.bar(x + (c_idx - 1) * width, means, width, yerr=stds,
                   label=channel, color=colors[c_idx], alpha=0.85, capsize=3)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(classes)
    ax_bar.set_ylabel("Pixel intensity (0–255)")
    ax_bar.set_title("Per-channel mean ± std by class", fontsize=13)
    ax_bar.legend(title="Channel")

    plt.tight_layout()
    plt.show()

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
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(shifts.keys(), shifts.values(), color="#c44e52")
    ax.set_ylabel("Euclidean distance (RGB mean)")
    ax.set_title(title)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    plt.tight_layout()
    plt.show()

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