"""
Encoder loading + frozen feature extraction for the pathology-vs-natural-image
pretraining comparison.

Transform contract (this was the bug in the previous version): every transform
returned by load_encoder takes an HWC uint8 NUMPY ARRAY, never a PIL Image.
timm's create_transform() starts with PIL-based Resize/CenterCrop and would
reject ndarray, so we build the transform explicitly from the resolved mean/std
instead. This also avoids any resize: tiles are already at the model's input
size, so there is nothing to interpolate.
"""

from pathlib import Path

import numpy as np
import torch
import timm
from timm.data import resolve_model_data_config
from torchvision import transforms as T


# VERIFY against https://huggingface.co/bioptimus/H-optimus-0 and against
# m.pretrained_cfg. These are from memory, not from the checkpoint.
H_OPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
H_OPTIMUS_STD = (0.211883, 0.230117, 0.177517)

TILE_SIZE = 224  # NCT-CRC / CRC-VAL tiles: 224x224 at 0.5 MPP

ENCODERS = {
    "h_optimus_0": {
        "source": "hf-hub:bioptimus/H-optimus-0",
        "kwargs": dict(init_values=1e-5, dynamic_img_size=False),
        "mean": H_OPTIMUS_MEAN,
        "std": H_OPTIMUS_STD,
        "note": "ViT-G/14, DINOv2-style SSL on H&E WSIs. Native 224.",
    },
    "dinov2_giant": {
         #overrides timm's registered 518. Without it the control
        # gets 1369 patches vs H-Optimus's 256 -- ~5x the compute and an
        # unmatched comparison. timm interpolates the position embeddings.
        "img_size": 224,
        "source": "vit_giant_patch14_dinov2.lvd142m",
        "kwargs": dict(img_size=TILE_SIZE),
        "mean": None,  # from timm pretrained_cfg
        "std": None,
        "note": "ViT-G/14, DINOv2 on LVD-142M. Run at 224, off its native 518.",
    },
    "vit_l16_in21k_ft_in1k": {
        "source": "vit_large_patch16_224.augreg_in21k_ft_in1k",
        "kwargs": dict(),
        "mean": None,
        "std": None,
        "note": "ViT-L/16 supervised. Weak baseline: varies arch, objective, domain.",
    },
}


def _model_input_size(model, data_config):
    """
    Actual spatial size the model was BUILT at.

    Not the same as data_config["input_size"], which comes from
    model.pretrained_cfg -- the checkpoint's registered config. timm does not
    write an img_size= override back into pretrained_cfg, so for DINOv2 the cfg
    keeps saying 518 while patch_embed is at 224. Trust patch_embed.
    """
    patch_embed = getattr(model, "patch_embed", None)
    size = getattr(patch_embed, "img_size", None)
    if size is None:
        return tuple(data_config["input_size"]), None, None
    size = tuple(size) if isinstance(size, (tuple, list)) else (size, size)
    return ((data_config["input_size"][0],) + size,
            getattr(patch_embed, "grid_size", None),
            getattr(patch_embed, "num_patches", None))


def load_encoder(name, device="cuda", dtype=torch.float32, verbose=True):
    """Returns (model, transform, info). Frozen, eval mode."""
    if name not in ENCODERS:
        raise KeyError(f"Unknown encoder {name!r}. Options: {list(ENCODERS)}")
    spec = ENCODERS[name]

    model = timm.create_model(spec["source"], pretrained=True, num_classes=0,
                              **spec["kwargs"])
    model = model.eval().to(device=device, dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)

    # mean/std are unaffected by any img_size override, so pretrained_cfg is
    # authoritative for those.
    data_config = resolve_model_data_config(model)
    mean = spec["mean"] if spec["mean"] is not None else data_config["mean"]
    std = spec["std"] if spec["std"] is not None else data_config["std"]

    input_size, grid_size, n_patches = _model_input_size(model, data_config)
    cfg_size = tuple(data_config["input_size"])

    if input_size[-1] != TILE_SIZE:
        raise ValueError(
            f"{name} was built at {input_size} but tiles are "
            f"{TILE_SIZE}x{TILE_SIZE}. Add img_size={TILE_SIZE} to this "
            f"encoder's kwargs in ENCODERS, or resize deliberately."
        )
    if cfg_size[-1] != input_size[-1] and verbose:
        print(f"  note: running off native resolution "
              f"(checkpoint registered at {cfg_size[-1]}, built at {input_size[-1]}); "
              f"position embeddings interpolated")

    # ToTensor: HWC uint8 -> CHW float in [0,1]. No resize, no PIL.
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    info = {
        "name": name,
        "source": spec["source"],
        "note": spec["note"],
        "embed_dim": getattr(model, "num_features", None),
        "n_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "mean": tuple(float(x) for x in mean),
        "std": tuple(float(x) for x in std),
        "input_size": input_size,
        "native_size": cfg_size,
        "grid_size": tuple(grid_size) if grid_size else None,
        "n_patches": n_patches,
        "mean_source": "hardcoded (VERIFY)" if spec["mean"] else "timm pretrained_cfg",
    }
    if verbose:
        for k, v in info.items():
            print(f"  {k:12s} {v}")
    return model, transform, info


def _autocast_dtype(device):
    if device.startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if device.startswith("cuda") else torch.float32


@torch.inference_mode()
def extract_features(model, transform, image_paths, load_fn, normalizer=None,
                     batch_size=32, device="cuda", verbose=True):
    """
    Frozen features for a list of tile paths.

    load_fn(path) -> HWC uint8 array (pass stain_norm.load_image_array).
    normalizer: optional fitted Macenko normalizer. Tiles where Macenko fails
        fall back to the RAW tile and are marked False in ok_mask, so features
        stay index-aligned with image_paths and labels.

    Returns (features (n, d) float32, ok_mask (n,) bool).
    """
    import stain_norm as sn

    amp_dtype = _autocast_dtype(device)
    feats, ok = [], []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        tensors = []
        for p in batch_paths:
            arr = load_fn(p)
            if normalizer is not None:
                norm = sn.normalize_image(p, normalizer)
                if norm is None:
                    ok.append(False)   # keep raw arr as fallback
                else:
                    arr = norm
                    ok.append(True)
            else:
                ok.append(True)

            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.shape[:2] != (TILE_SIZE, TILE_SIZE):
                raise ValueError(f"{p}: expected {TILE_SIZE}x{TILE_SIZE}, got {arr.shape}")
            tensors.append(transform(arr))

        batch = torch.stack(tensors).to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                            enabled=device.startswith("cuda")):
            out = model(batch)
        feats.append(out.float().cpu().numpy())

        if verbose and (start // batch_size) % 20 == 0:
            print(f"  {start + len(batch_paths)}/{len(image_paths)}")

    return np.concatenate(feats, axis=0), np.array(ok)


def save_features(path, features, labels, ok_mask, info):
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=features, labels=np.asarray(labels),
             ok_mask=ok_mask, encoder=info["name"], source=info["source"],
             embed_dim=info["embed_dim"], mean=np.asarray(info["mean"]),
             std=np.asarray(info["std"]))
    return path