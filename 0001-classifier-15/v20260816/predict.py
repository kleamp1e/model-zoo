"""Minimal inference sample for the classifier-15 ONNX models."""

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime
from PIL import Image


def load_manifest(model_dir: Path) -> dict:
    with (model_dir / "manifest.json").open(encoding="utf-8") as file:
        return json.load(file)


def find_output(model_info: dict, name: str) -> dict:
    for output in model_info["outputs"]:
        if output["name"] == name:
            return output
    raise KeyError(f"output not found: {name}")


def load_rgb_image(image_path: Path) -> np.ndarray:
    """Load an image as a uint8 NHWC array with shape (1, height, width, 3)."""
    with Image.open(image_path) as image:
        rgb_image = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return rgb_image[np.newaxis, :, :, :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="path to the input image file")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("."),
        help="directory containing manifest.json and the ONNX files (default: current directory)",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.model_dir)
    preprocessor_info = manifest["preprocessor"]
    classifier_info = manifest["classifier"]
    labels = find_output(classifier_info, "probabilities")["labels"]

    preprocessor = onnxruntime.InferenceSession(
        args.model_dir / preprocessor_info["onnx_file"]["name"]
    )
    classifier = onnxruntime.InferenceSession(
        args.model_dir / classifier_info["onnx_file"]["name"]
    )

    rgb_image = load_rgb_image(args.image)
    images = cast(np.ndarray, preprocessor.run(["images"], {"rgb_image": rgb_image})[0])
    probabilities = cast(
        np.ndarray, classifier.run(["probabilities"], {"images": images})[0]
    )

    name_width = max(len(label["name"]) for label in labels)
    for label, probability in zip(labels, probabilities[0], strict=True):
        threshold = label["recommended_threshold"]
        judgement = "positive" if probability >= threshold else "negative"
        print(
            f"{label['name']:<{name_width}}  "
            f"probability={probability:.6f}  "
            f"threshold={threshold:.6f}  "
            f"{judgement}"
        )


if __name__ == "__main__":
    main()
