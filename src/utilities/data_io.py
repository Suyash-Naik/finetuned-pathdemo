from pathlib import Path
import zipfile
from bioio import BioImage
import bioio_tifffile
import random
import shutil
import numpy as np


def extract_zip(zip_path, extract_to):
    """Extracts a zip file to the specified directory, skipping if already extracted."""
    target = Path(extract_to) / zip_path.stem
    if target.exists():
        print(f"Skipping {zip_path.name}, already extracted.")
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"Extracted {zip_path.name} to {target}")

def count_patches_per_class(root_path):
    """
    Counts .tif patches per class subfolder under root_path.
    Returns dict: {class_name: count}
    """
    class_dirs = sorted([d for d in Path(root_path).iterdir() if d.is_dir()])
    counts = {}
    for class_dir in class_dirs:
        counts[class_dir.name] = sum(1 for _ in class_dir.glob("*.tif"))
    return counts

def get_image_metadata(image_path, reader=bioio_tifffile.reader.Reader, verbose=False):
    """
    Loads a single image and returns its key metadata as a dict.
    Set verbose=True to also print a human-readable summary.
    """
    img = BioImage(image_path, reader=reader)

    metadata = {
        "path": str(image_path),
        "dimension_order": img.dims.order,
        "shape": img.shape,
        "channel_names": img.channel_names,
        "scenes": img.scenes,
        "current_scene": img.current_scene,
        "pixel_size_x": img.physical_pixel_sizes.X,
        "pixel_size_y": img.physical_pixel_sizes.Y,
        "pixel_size_z": img.physical_pixel_sizes.Z,
    }

    if verbose:
        for key, value in metadata.items():
            print(f"{key}: {value}")

    return metadata

def summarize_metadata_by_class(root_path, reader=bioio_tifffile.reader.Reader, n_samples=1):
    """
    Walks each class subfolder under root_path (e.g. ADI, BACK, DEB, ...),
    grabs n_samples .tif files per class, and collects their metadata.
    Returns a dict keyed by class name -> list of metadata dicts.
    """
    class_dirs = sorted([d for d in Path(root_path).iterdir() if d.is_dir()])

    if not class_dirs:
        print(f"No subfolders found under {root_path}")
        return {}

    results = {}
    for class_dir in class_dirs:
        tif_files = list(class_dir.glob("*.tif"))
        if not tif_files:
            print(f"Warning: no .tif files found in {class_dir.name}")
            continue

        sample_files = tif_files[:n_samples]
        results[class_dir.name] = [
            get_image_metadata(f, reader=reader, verbose=False) for f in sample_files
        ]

    return results

def print_class_metadata_summary(results):
    """Prints a compact one-line-per-class summary from summarize_metadata_by_class output."""
    for class_name, meta_list in results.items():
        first = meta_list[0]
        print(f"{class_name:6s} | n_files_sampled={len(meta_list)} | "
              f"shape={first['shape']} | dim_order={first['dimension_order']} | pixel_size ={first['pixel_size_x']}, {first['pixel_size_y']}")
        
def export_sample_subset(root_path, output_dir, n_samples=4, seed=42):
    """One-time export: copies the fixed random sample used for visualization
    into a small folder that's cheap enough to commit to the repo."""
    class_dirs = sorted([d for d in Path(root_path).iterdir() if d.is_dir()])
    rng = random.Random(seed)
    output_dir = Path(output_dir)

    for class_dir in class_dirs:
        tif_files = list(class_dir.glob("*.tif"))
        sample_files = rng.sample(tif_files, min(n_samples, len(tif_files)))
        out_class_dir = output_dir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)
        for f in sample_files:
            shutil.copy(f, out_class_dir / f.name)

def compute_channel_stats(root_path, n_samples_per_class=200, reader=bioio_tifffile.reader.Reader, seed=69): #Nice
    """
    Samples n_samples_per_class images per class and computes per-channel
    (R, G, B) mean and std, both per-class and overall.
    Returns a dict: {class_name: {"mean": [R,G,B], "std": [R,G,B], "n": int}, ...}
    plus an "overall" key aggregating across all sampled classes.
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
        for f in sample_files:
            img = BioImage(f, reader=reader).get_image_data("YXS")
            pixels.append(img.reshape(-1, img.shape[-1]))  # flatten spatial dims, keep channels
        pixels = np.concatenate(pixels, axis=0)

        stats[class_dir.name] = {
            "mean": pixels.mean(axis=0).tolist(),
            "std": pixels.std(axis=0).tolist(),
            "n": len(sample_files),
        }
        all_pixels.append(pixels)

    overall = np.concatenate(all_pixels, axis=0)
    stats["overall"] = {
        "mean": overall.mean(axis=0).tolist(),
        "std": overall.std(axis=0).tolist(),
        "n": sum(v["n"] for k, v in stats.items()),
    }
    return stats
