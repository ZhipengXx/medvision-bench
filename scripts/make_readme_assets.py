"""Create original vector showcase artwork from the published numeric results.

The decorative imagery is synthetic. No participant image or volume is embedded.
The script uses the standard library and keeps SVG lettering editable.
"""
from __future__ import annotations

import csv
import html
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
INK = "#152D42"
MUTED = "#536D80"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"


def text(x, y, value, size=20, color=INK, weight=400, anchor="start"):
    return (f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" '
            'font-family="Arial, Helvetica, sans-serif">'
            f'{html.escape(str(value))}</text>')


def rect(x, y, width, height, fill, radius=0, stroke="none"):
    return (f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}"/>')


def line(x1, y1, x2, y2, color="#DDE7ED", width=1):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{width}"/>')


def save(name, width, height, title, description, pieces):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'width="{width}" height="{height}" role="img" aria-labelledby="title desc">'
           f'<title id="title">{html.escape(title)}</title>'
           f'<desc id="desc">{html.escape(description)}</desc>'
           + "".join(pieces) + "</svg>\n")
    (ASSETS / name).write_text(svg, encoding="utf-8")


def hero():
    p = [rect(0, 0, 1280, 400, "#0B1B2E", 24)]
    p += [rect(1080, -100, 320, 650, "#102D47", 150),
          rect(977, -60, 230, 560, "#11344F", 115),
          text(64, 65, "OPEN RESEARCH  /  MEDICAL IMAGING", 15, "#74DCCB", 700),
          text(62, 132, "MedVision Bench", 58, "#FFFFFF", 700),
          text(65, 177, "Classification. Segmentation. Evidence.", 27, "#D6E6F0"),
          text(65, 218, "An auditable benchmark for loss functions and spatial context.", 19, "#9FB6C9")]
    for x, value, label in [(65, "7", "skin lesion classes"),
                            (287, "5", "primary models"),
                            (509, "2", "imaging tasks")]:
        p += [rect(x, 273, 196, 82, "#132E45", 12, "#26465D"),
              text(x + 20, 310, value, 30, "#FFFFFF", 700),
              text(x + 20, 337, label, 15, "#AFC6D7")]
    # Original abstract neural-network illustration: no medical input imagery.
    for x, ys in [(831, [124, 197, 270]), (909, [100, 165, 230, 295]),
                   (987, [130, 205, 280])]:
        next_ys = {831: [100, 165, 230, 295], 909: [130, 205, 280]}.get(x, [])
        for y in ys:
            for ny in next_ys:
                p += [line(x, y, x + 78, ny, "#245772", 1.4)]
    for x, ys in [(831, [124, 197, 270]), (909, [100, 165, 230, 295]),
                   (987, [130, 205, 280])]:
        for y in ys:
            p += [f'<circle cx="{x}" cy="{y}" r="8" fill="#67D6C8"/>']
    for i, h in enumerate([52, 86, 118, 94, 145]):
        p += [rect(1060 + 27 * i, 309 - h, 17, h, ["#65D8C8", "#72B9E3"][i % 2], 4)]
    p += [text(830, 354, "FROM PREDICTIONS TO UNDERSTANDING", 11, "#92B6CC", 700)]
    save("hero.svg", 1280, 400, "MedVision Bench",
         "Medical image classification and segmentation research. Seven skin lesion classes, "
         "five primary models, and two imaging tasks. Decorative network illustration is synthetic.", p)


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def highlights():
    cls = {r["run_id"]: r for r in rows(ROOT / "results/classification/summary.csv")}
    seg = {r["run_id"]: r for r in rows(ROOT / "results/segmentation/summary.csv")}
    ce, wce = cls["cls_r50_ce"], cls["cls_r50_wce"]
    accuracy = float(ce["accuracy"]) * 100
    recall_delta = (float(wce["macro_recall"]) - float(ce["macro_recall"])) * 100
    dice = float(seg["seg_unet3d"]["dice_mean"]) * 100
    p = [rect(0, 0, 1280, 220, "#F5F9FC", 20)]
    cards = [(20, BLUE, "CLASSIFICATION", f"{accuracy:.2f}%", "CE test accuracy", "1,512 held-out images"),
             (440, ORANGE, "LOSS TRADE-OFF", f"+{recall_delta:.2f} pp", "Weighted CE macro recall vs CE", "Accuracy decreases by 3.37 pp"),
             (860, GREEN, "VOLUMETRIC SEGMENTATION", f"{dice:.2f}%", "3D U-Net mean Dice", "20 held-out volumes")]
    for x, color, label, value, detail, foot in cards:
        p += [rect(x, 20, 400, 180, "#FFFFFF", 14, "#E0E9EF"),
              rect(x + 20, 41, 4, 18, color, 2),
              text(x + 36, 55, label, 13, MUTED, 700),
              text(x + 25, 109, value, 40, color, 700),
              text(x + 25, 144, detail, 16, INK),
              text(x + 25, 176, foot, 14, MUTED)]
    save("highlights.svg", 1280, 220, "Primary results at a glance",
         f"CE classification accuracy {accuracy:.2f} percent. Weighted CE macro recall "
         f"improves by {recall_delta:.2f} percentage points while accuracy decreases by "
         f"3.37 percentage points. 3D U-Net mean Dice {dice:.2f} percent.", p)


def pipeline():
    p = [rect(0, 0, 1280, 562, "#F7FAFC", 20),
         text(38, 53, "Two tasks. One evaluation discipline.", 29, INK, 700),
         text(38, 88, "Training learns parameters; validation selects checkpoints; test estimates performance.", 17, MUTED)]
    headers = [(38, "01  INPUT & SPLIT"), (349, "02  INDEPENDENT TRAINING"),
               (660, "03  VALIDATION SELECTION"), (971, "04  FROZEN TEST")]
    for x, label in headers:
        p += [text(x, 132, label, 13, MUTED, 700)]
    cards = [
        (38, 151, BLUE, "Dermoscopy", ["ISIC 2018 · seven classes", "Train / validation / test", "10,015 / 193 / 1,512 images"]),
        (349, 151, BLUE, "ResNet-50", ["Shared ImageNet initialization", "CE · weighted CE · focal", "Independent optimization"]),
        (660, 151, BLUE, "Validation accuracy", ["Macro F1 breaks ties", "Early stopping on validation", "No test-driven selection"]),
        (971, 151, BLUE, "Classification", ["Argmax predictions", "Per-class and macro metrics", "Paired bootstrap contrasts"]),
        (38, 327, GREEN, "Left atrial MRI", ["Case-separated HDF5 volumes", "Train / validation / test", "11 / 3 / 20 volumes"]),
        (349, 327, GREEN, "U-Net", ["2D slices · 3D patches", "Image-only normalization", "CE + foreground soft Dice"]),
        (660, 327, GREEN, "Validation Dice", ["Whole-volume inference", "Same held-out cases", "Checkpoint hash recorded"]),
        (971, 327, GREEN, "Segmentation", ["Gaussian window fusion", "Dice · Jaccard · ASD · HD95", "Every supplied test voxel"]),
    ]
    for x, y, color, title, bullets in cards:
        p += [rect(x, y, 273, 144, "#FFFFFF", 13, "#D8E4EC"),
              rect(x, y, 6, 144, color, 3), text(x + 22, y + 34, title, 22, INK, 700)]
        for i, item in enumerate(bullets):
            p += [text(x + 22, y + 67 + i * 24, item, 14, MUTED)]
    for y, color in [(223, BLUE), (399, GREEN)]:
        for x in [315, 626, 937]:
            p += [f'<path d="M{x},{y} h27 m-7,-6 7,6 -7,6" '
                  f'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round"/>']
    p += [rect(38, 493, 1206, 41, "#EAF1F6", 10),
          text(60, 520, "PRIMARY PROTOCOL  ·  Single seed 42  ·  Test partitions remain outside fitting and model selection", 15, MUTED)]
    save("pipeline.svg", 1280, 562, "Primary benchmark methodology",
         "Two independent pipelines with training, validation checkpoint selection and frozen test "
         "evaluation. ResNet-50 compares CE, weighted CE and focal loss; U-Net compares 2D and 3D "
         "context. Diagram is original vector artwork without participant data.", p)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    hero()
    highlights()
    pipeline()
    for name in ["classification_main", "segmentation_main"]:
        shutil.copyfile(ROOT / "results" / "figures" / f"{name}.png", ASSETS / f"{name}.png")
    print("Created 3 original SVG assets and copied 2 quantitative figures to docs/assets.")


if __name__ == "__main__":
    main()
