"""Run pose estimation and six attribute classifiers on one image."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def session(model_dir: Path, filename: str) -> ort.InferenceSession:
    path = model_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def pose_input(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    gain = min(640 / height, 640 / width)
    resized_width, resized_height = round(width * gain), round(height * gain)
    resized = cv2.resize(image, (resized_width, resized_height))
    pad_x = round((640 - resized_width) / 2 - 0.1)
    pad_y = round((640 - resized_height) / 2 - 0.1)
    padded = cv2.copyMakeBorder(
        resized,
        pad_y,
        640 - resized_height - pad_y,
        pad_x,
        640 - resized_width - pad_x,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    tensor = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    return tensor, gain, pad_x, pad_y


def pose_detections(
    output: np.ndarray,
    image_shape: tuple[int, ...],
    gain: float,
    pad_x: int,
    pad_y: int,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if output.shape != (1, 9, 8400):
        raise ValueError(f"Unexpected pose output shape: {output.shape}")
    rows = output[0].T
    rows = rows[rows[:, 4] > confidence]
    boxes = np.empty((len(rows), 4), dtype=np.float32)
    boxes[:, :2] = rows[:, :2] - rows[:, 2:4] / 2
    boxes[:, 2:] = rows[:, :2] + rows[:, 2:4] / 2
    scores = rows[:, 4]
    order = np.argsort(-scores, kind="stable")[:30000]
    areas = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    kept = []
    while len(order) and len(kept) < 300:
        index = order[0]
        kept.append(index)
        rest = order[1:]
        overlap_min = np.maximum(boxes[index, :2], boxes[rest, :2])
        overlap_max = np.minimum(boxes[index, 2:], boxes[rest, 2:])
        overlap = np.maximum(overlap_max - overlap_min, 0)
        intersection = overlap[:, 0] * overlap[:, 1]
        iou = intersection / (areas[index] + areas[rest] - intersection + 1e-7)
        order = rest[iou <= 0.7]

    boxes = boxes[kept]
    points = rows[kept, 5:9].reshape(-1, 2, 2).copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / gain
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / gain
    points[:, :, 0] = (points[:, :, 0] - pad_x) / gain
    points[:, :, 1] = (points[:, :, 1] - pad_y) / gain
    height, width = image_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    points[:, :, 0] = points[:, :, 0].clip(0, width)
    points[:, :, 1] = points[:, :, 1].clip(0, height)
    return boxes, scores[kept], points


def predict(image_path: Path, model_dir: Path, confidence: float) -> dict:
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    tensor, gain, pad_x, pad_y = pose_input(image)
    pose = session(model_dir, manifest["pose"]["onnx_file"]["name"])
    output = pose.run(["output0"], {"images": tensor})[0]
    boxes, scores, points = pose_detections(
        output, image.shape, gain, pad_x, pad_y, confidence
    )
    if not len(boxes):
        return {"detections": []}

    preprocess = session(model_dir, manifest["preprocessor"]["onnx_file"]["name"])
    crops = preprocess.run(
        ["crops"],
        {
            "image": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "boxes": boxes,
            "points": points,
        },
    )[0]
    if crops.shape != (len(boxes), 3, 224, 224):
        raise ValueError(f"Unexpected preprocessor output shape: {crops.shape}")
    classifier = session(model_dir, manifest["classifier"]["onnx_file"]["name"])
    probabilities = classifier.run(["probabilities"], {"image": crops})[0]
    attributes = manifest["classifier"]["outputs"][0]["labels"]
    if probabilities.shape != (len(boxes), len(attributes)):
        raise ValueError(f"Unexpected classifier output shape: {probabilities.shape}")
    return {
        "detections": [
            {
                "box": list(map(float, box)),
                "confidence": float(score),
                "points": point.astype(float).tolist(),
                "attributes": dict(zip(attributes, map(float, values), strict=True)),
            }
            for box, score, point, values in zip(
                boxes, scores, points, probabilities, strict=True
            )
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument(
        "--model-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    try:
        result = predict(args.image, args.model_dir, args.confidence)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
