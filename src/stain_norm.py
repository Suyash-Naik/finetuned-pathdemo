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
    Returns the fitted normalizer — reuse it across every normalize_image() call
    rather than re-fitting per image.
    """
    reference_array = load_image_array(reference_path, reader=reader)
    reference_tensor = _to_tensor_255(reference_array)

    normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
    normalizer.fit(reference_tensor)
    return normalizer


def normalize_image(image_path, normalizer, reader=bioio_tifffile.reader.Reader):
    """
    Applies a fitted Macenko normalizer to a single image.
    Returns normalized image as HxWxC uint8, or None if normalization fails
    (this happens on near-white/background patches with too little stain
    signal for Macenko's optical-density decomposition to work on).
    """
    image_array = load_image_array(image_path, reader=reader)
    image_tensor = _to_tensor_255(image_array)

    try:
        norm, _, _ = normalizer.normalize(I=image_tensor, stains=True)
    except Exception as e:
        print(f"Normalization failed for {image_path}: {e}")
        return None

    return norm.numpy().astype(np.uint8)


def sanity_check_normalization(root_path, normalizer, n_samples=6,
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
        title="Macenko normalization — sanity check (raw vs. normalized)",
        width=n_samples * 180 + 60, height=420,
        margin=dict(t=80, l=60, r=20, b=20),
    )
    return fig


def plot_normalized_class_grid(root_path, normalizer, n_samples=4,
                                reader=bioio_tifffile.reader.Reader, seed=42):
    """
    Same layout as plot_class_sample_grid, but shows Macenko-normalized tiles.
    Uses the same seed as the original unnormalized grid so the two figures
    show the exact same underlying patches, for a direct before/after comparison.
    """
    class_dirs = sorted([d for d in Path(root_path).iterdir() if d.is_dir()])
    rng = random.Random(seed)

    n_classes = len(class_dirs)
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(n_samples * 2.2, n_classes * 2.2))
    fig.suptitle("Sample patches per tissue class, Macenko-normalized", fontsize=20)

    for row, class_dir in enumerate(class_dirs):
        tif_files = list(class_dir.glob("*.tif"))
        if not tif_files:
            print(f"Warning: no .tif files found in {class_dir.name}")
            continue

        sample_files = rng.sample(tif_files, min(n_samples, len(tif_files)))

        for col in range(n_samples):
            ax = axes[row, col]
            if col < len(sample_files):
                norm = normalize_image(sample_files[col], normalizer, reader=reader)
                if norm is not None:
                    ax.imshow(norm)
            ax.axis("off")
            ax.set_title(class_dir.name, fontsize=16)

    plt.tight_layout()
    plt.show()