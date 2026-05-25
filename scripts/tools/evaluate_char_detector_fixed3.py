#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import shutil
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset" / "char_detector_yolo"
MODEL = ROOT / "runs" / "detect" / "runs" / "detect" / "dataset" / "char_detector_yolo" / "runs" / "char_detector_yolov8n_clean" / "weights" / "best.pt"


def box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def yolo_to_xyxy(line: str, size: tuple[int, int]) -> tuple[float, float, float, float]:
    _, cx, cy, w, h = [float(x) for x in line.split()]
    width, height = size
    bw = w * width
    bh = h * height
    x = cx * width
    y = cy * height
    return x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / (box_area(a) + box_area(b) - inter + 1e-6)


def center_error(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)


def best_match(gt: list[tuple[float, float, float, float]], pred: list[tuple[float, float, float, float]]) -> tuple[tuple[int, ...], list[float]]:
    best_perm = ()
    best_scores: list[float] = []
    best_total = -1.0
    for perm in permutations(range(len(pred)), len(gt)):
        scores = [iou(gt[i], pred[perm[i]]) for i in range(len(gt))]
        total = float(sum(scores))
        if total > best_total:
            best_total = total
            best_perm = perm
            best_scores = scores
    return best_perm, best_scores


def select_fixed3(
    boxes: list[tuple[float, float, float, float]],
    confs: list[float],
    image_size: tuple[int, int],
) -> tuple[list[tuple[float, float, float, float]], list[float], str]:
    width, height = image_size
    candidates = []
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        area = bw * bh
        if bw < 22 or bh < 22:
            continue
        if area < 550:
            continue
        if bw > width * 0.38 or bh > height * 0.38:
            continue
        ratio = bw / max(1.0, bh)
        if ratio < 0.35 or ratio > 2.6:
            continue
        candidates.append((box, conf, area))

    if len(candidates) >= 4:
        areas = np.array([item[2] for item in candidates], dtype=np.float32)
        median_area = float(np.median(areas))
        candidates = [
            item
            for item in candidates
            if item[2] >= median_area * 0.35 and item[2] <= median_area * 2.8
        ]

    if len(candidates) < 3:
        return [item[0] for item in candidates], [item[1] for item in candidates], "FAIL_LT3"

    ranked = sorted(candidates, key=lambda item: item[1] * (item[2] ** 0.5), reverse=True)
    selected = ranked[:3]
    return [item[0] for item in selected], [item[1] for item in selected], "FIXED3"


def draw_preview(image: Image.Image, pred: list[tuple[float, float, float, float]], confs: list[float], out: Path, gt=None, status="") -> None:
    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    if gt:
        for idx, box in enumerate(gt, start=1):
            draw.rectangle(box, outline="lime", width=3)
            draw.text((box[0], max(0, box[1] - 16)), f"GT{idx}", fill="lime")
    for idx, (box, conf) in enumerate(zip(pred, confs), start=1):
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], max(0, box[1] - 16)), f"P{idx} {conf:.2f}", fill="red")
    draw.text((8, 8), f"{status} pred={len(pred)}", fill="yellow")
    marked.save(out)


def eval_val(model: YOLO, model_path: Path) -> None:
    out_dir = DATASET / "fixed3_eval_val"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "preview").mkdir(parents=True, exist_ok=True)
    (out_dir / "failed").mkdir(parents=True, exist_ok=True)
    rows = []
    passes = 0
    lt3 = 0
    fail_iou = 0
    min_ious = []
    errors = []
    for image_path in sorted((DATASET / "images" / "val").glob("*.png")):
        image = Image.open(image_path).convert("RGB")
        label_path = DATASET / "labels" / "val" / f"{image_path.stem}.txt"
        gt = [yolo_to_xyxy(line, image.size) for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = model.predict(source=image, imgsz=448, conf=0.15, iou=0.5, max_det=10, verbose=False)[0]
        raw_boxes, raw_confs = [], []
        if result.boxes is not None:
            for box in result.boxes:
                raw_boxes.append(tuple(float(x) for x in box.xyxy[0].tolist()))
                raw_confs.append(float(box.conf[0]))
        pred, confs, select_status = select_fixed3(raw_boxes, raw_confs, image.size)
        if len(pred) < 3:
            status = "FAIL_LT3"
            lt3 += 1
            matched = ()
            ious = []
            err = []
        else:
            matched, ious = best_match(gt, pred)
            err = [center_error(gt[i], pred[matched[i]]) for i in range(3)]
            if min(ious) >= 0.5:
                status = "PASS"
                passes += 1
            else:
                status = "FAIL_IOU"
                fail_iou += 1
            min_ious.append(min(ious))
            errors.extend(err)
        draw_preview(image, pred, confs, out_dir / "preview" / image_path.name, gt=gt, status=status)
        if status != "PASS":
            shutil.copy2(out_dir / "preview" / image_path.name, out_dir / "failed" / image_path.name)
        rows.append(
            {
                "file": image_path.name,
                "raw_count": len(raw_boxes),
                "final_count": len(pred),
                "status": status,
                "select_status": select_status,
                "min_iou": f"{min(ious):.4f}" if ious else "",
                "mean_center_error": f"{np.mean(err):.2f}" if err else "",
                "confs": " ".join(f"{c:.3f}" for c in confs),
            }
        )
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    total = len(rows)
    print("[val]")
    print(f"model={model_path}")
    print(f"pass={passes}/{total}={passes / total:.1%}")
    print(f"fail_lt3={lt3}")
    print(f"fail_iou={fail_iou}")
    if min_ious:
        print(f"min_iou_avg={np.mean(min_ious):.4f}")
    if errors:
        print(f"center_error_avg_px={np.mean(errors):.2f}")
    print(f"out={out_dir}")


def eval_excluded(model: YOLO, model_path: Path) -> None:
    excluded_script = ROOT / "scripts" / "tools" / "test_excluded_char_images.py"
    # Inline the same source discovery to avoid duplicating preview style.
    from test_excluded_char_images import current_trainval_names, OLD_SOURCE, NEW_SOURCE

    out_dir = DATASET / "fixed3_eval_excluded"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "all").mkdir(parents=True, exist_ok=True)
    (out_dir / "not3").mkdir(parents=True, exist_ok=True)
    used = current_trainval_names()
    rows = []
    for source_name, source_dir in [("old310", OLD_SOURCE), ("new102", NEW_SOURCE)]:
        for image_path in sorted(source_dir.glob("*.png")):
            if image_path.name in used:
                continue
            image = Image.open(image_path).convert("RGB")
            result = model.predict(source=image, imgsz=448, conf=0.15, iou=0.5, max_det=10, verbose=False)[0]
            raw_boxes, raw_confs = [], []
            if result.boxes is not None:
                for box in result.boxes:
                    raw_boxes.append(tuple(float(x) for x in box.xyxy[0].tolist()))
                    raw_confs.append(float(box.conf[0]))
            pred, confs, select_status = select_fixed3(raw_boxes, raw_confs, image.size)
            status = "pred3" if len(pred) == 3 else "not3"
            out_name = f"{source_name}__{image_path.name}"
            draw_preview(image, pred, confs, out_dir / "all" / out_name, status=status)
            if status != "pred3":
                shutil.copy2(out_dir / "all" / out_name, out_dir / "not3" / out_name)
            rows.append(
                {
                    "source": source_name,
                    "file": image_path.name,
                    "raw_count": len(raw_boxes),
                    "final_count": len(pred),
                    "status": status,
                    "select_status": select_status,
                    "confs": " ".join(f"{c:.3f}" for c in confs),
                }
            )
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    total = len(rows)
    pred3 = sum(1 for row in rows if row["status"] == "pred3")
    print("[excluded]")
    print(f"model={model_path}")
    print(f"pred3={pred3}/{total}={pred3 / total:.1%}" if total else "pred3=0/0")
    print(f"not3={total - pred3}")
    print(f"out={out_dir}")


def main() -> int:
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else MODEL
    model = YOLO(str(model_path))
    eval_val(model, model_path)
    eval_excluded(model, model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
