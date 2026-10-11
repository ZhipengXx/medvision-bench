"""Compute real CE Grad-CAMs and preserve arrays for reproducible restyling.

Case selection, model target and input geometry follow the original experiment.
Rendering is separate so later presentation changes do not require inference.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from src.classification.data import CLASSES, transform
from src.classification.model import build_model
from src.common.runtime import ROOT, load_config, save_json, sha256


SELECTION_RULE = "Highest confidence with diverse GT classes, separately correct/incorrect"
INPUT_GEOMETRY = "Whole image resized to 224x224; display matches deterministic network input"


def select_cases(rows):
    """Keep the original two-correct/two-incorrect, diverse-GT rule unchanged."""
    selected = []
    for correct in [True, False]:
        candidates = [r for r in rows if (r["gt"] == r["pred"]) == correct]
        candidates.sort(key=lambda r: float(r["p_" + CLASSES[int(r["pred"])]]), reverse=True)
        chosen = []
        for row in candidates:
            if not chosen or row["gt"] not in [x["gt"] for x in chosen]:
                chosen.append(row)
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            chosen = candidates[:2]
        if len(chosen) != 2:
            raise ValueError("Need two correct and two wrong real test cases")
        selected += chosen
    return selected


def cache_is_current(path, checkpoint_hash, image_hash, pred):
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                data["checkpoint_sha256"].item() == checkpoint_hash
                and data["source_image_sha256"].item() == image_hash
                and data["target_index"].item() == pred
                and data["input_geometry"].item() == INPUT_GEOMETRY
                and data["rgb"].shape == (224, 224, 3)
                and data["heat"].shape == (224, 224)
                and np.isfinite(data["heat"]).all()
            )
    except (KeyError, ValueError, OSError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4, help="CPU intra-op threads")
    parser.add_argument("--force", action="store_true", help="Recompute cached CAMs")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    results = args.results.resolve()
    prediction_path = results / "classification/cls_r50_ce/predictions.csv"
    with prediction_path.open(newline="") as handle:
        selected = select_cases(list(csv.DictReader(handle)))
    split_path = ROOT / "splits/isic_split.json"
    split = json.loads(split_path.read_text())
    paths = {r["image_id"]: r["path"] for r in split["test"]}
    image_root = ROOT / load_config(ROOT / "configs/data.yaml")["classification"]["root"]
    checkpoint_path = ROOT / "outputs/cls_r50_ce/best.pt"
    checkpoint_hash = sha256(checkpoint_path)
    assets_dir = results / "gradcam_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    model = None
    state = {}
    hook_handle = None
    records, assets = [], []
    try:
        for row in selected:
            image_id = row["image_id"]
            gt, pred = int(row["gt"]), int(row["pred"])
            conf = float(row["p_" + CLASSES[pred]])
            source_path = image_root / paths[image_id]
            image_hash = sha256(source_path)
            asset_path = assets_dir / f"{image_id}.npz"
            cached = not args.force and cache_is_current(asset_path, checkpoint_hash, image_hash, pred)
            if not cached:
                if model is None:
                    model = build_model(False).to(device).eval()
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    model.load_state_dict(checkpoint["model"])
                    del checkpoint

                    def capture_activation(module, inputs, output):
                        state["activation"] = output
                        output.retain_grad()

                    hook_handle = model.layer4[-1].register_forward_hook(capture_activation)
                with Image.open(source_path) as source:
                    im = source.convert("RGB")
                rgb = np.array(im.resize((224, 224), Image.Resampling.BILINEAR))
                x = transform(False)(im)[None].to(device).requires_grad_(True)
                model.zero_grad(set_to_none=True)
                # Grad-CAM requires gradients in eval mode; never use inference_mode.
                logits = model(x)
                actual_pred = int(logits.argmax(dim=1).item())
                actual_conf = float(logits.softmax(dim=1)[0, pred].detach().cpu())
                if actual_pred != pred or abs(actual_conf - conf) > 1e-4:
                    raise ValueError(f"Checkpoint prediction disagrees with recorded test row: {image_id}")
                logits[0, pred].backward()
                activation = state["activation"]
                weights = activation.grad.mean((2, 3), keepdim=True)
                heat = (weights * activation).sum(1, keepdim=True).relu()
                heat = F.interpolate(heat, (224, 224), mode="bilinear", align_corners=False)
                heat = heat[0, 0].detach().cpu().numpy()
                heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
                np.savez_compressed(
                    asset_path,
                    rgb=rgb,
                    heat=heat.astype(np.float32, copy=False),
                    target_index=np.asarray(pred),
                    checkpoint_sha256=np.asarray(checkpoint_hash),
                    source_image_sha256=np.asarray(image_hash),
                    input_geometry=np.asarray(INPUT_GEOMETRY),
                    recomputed_confidence=np.asarray(actual_conf),
                    device=np.asarray(str(device)),
                )
            records.append(dict(
                image_id=image_id, gt=CLASSES[gt], pred=CLASSES[pred],
                confidence=conf, correct=gt == pred,
                selection_rule=SELECTION_RULE, cam_target=CLASSES[pred],
                input_geometry=INPUT_GEOMETRY,
            ))
            assets.append(dict(image_id=image_id, path=str(asset_path.relative_to(results)),
                               sha256=sha256(asset_path), source_image_sha256=image_hash))
            print(f"{image_id}: {'cached' if cached else 'computed'} predicted-class CAM", flush=True)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    # Existing consumers continue to read the unchanged `cases` schema.
    save_json(results / "gradcam_cases.json", dict(cases=records, provenance=dict(
        checkpoint="outputs/cls_r50_ce/best.pt", checkpoint_sha256=checkpoint_hash,
        predictions_sha256=sha256(prediction_path), split_sha256=sha256(split_path),
        layer="layer4[-1]", method="ReLU of gradient-weighted activations; bilinear resize",
        normalization="Per-image min-max normalization, epsilon 1e-8",
        assets=assets,
    )))
    from restyle_gradcam import render_gradcam
    outputs = render_gradcam(results)
    print("Saved " + ", ".join(str(p) for p in outputs), flush=True)


if __name__ == "__main__":
    main()
