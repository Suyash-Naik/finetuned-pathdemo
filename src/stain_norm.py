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
    Returns the fitted normalizer 
    reuse across every normalize_image() call
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

def plot_grid_from_data_plotly(grid_data, title="Sample patches per tissue class, Macenko-normalized"):
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