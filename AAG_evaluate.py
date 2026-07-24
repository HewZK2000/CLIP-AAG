#!/usr/bin/env python3
"""Load a trained AAG generator and classify test image embeddings.

AAG operates in a 512-dimensional embedding space. ``--test-image`` therefore
accepts image embeddings, and ``--test-text`` accepts labeled candidate text
embeddings produced by the same encoder used to prepare the training dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


EMBEDDING_DIM = 512


class Generator(nn.Module):
    """AAG text-to-visual-prototype generator."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 256),
            nn.LeakyReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(inplace=True),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(256, EMBEDDING_DIM),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(input_tensor)
        enc2 = self.encoder2(enc1)
        dec1 = self.decoder1(enc2) + enc1
        return self.decoder2(dec1) + input_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict the most likely text label for one or more 512-dimensional "
            "test image embeddings."
        )
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Path to a wgan_prototype_*.pth generator checkpoint.",
    )
    parser.add_argument(
        "--test-image",
        type=Path,
        required=True,
        help=(
            "Path to test image embedding(s): .npy, .npz, .pt, .pth, .json, "
            ".csv, or .txt."
        ),
    )
    parser.add_argument(
        "--test-text",
        type=Path,
        required=True,
        help=(
            "Path to labeled candidate text embeddings: .pt, .pth, .json, "
            "or .npz."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of ranked labels to return for each image.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device. 'auto' uses CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path at which to save the prediction results as JSON.",
    )

    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")
    return args


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def load_generator(
    checkpoint_path: Path,
    device: torch.device,
) -> Generator:
    """Load a generator state dictionary saved by AAG_training.py."""

    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Expected the checkpoint to contain a state dictionary.")

    for state_key in ("state_dict", "generator_state_dict", "model_state_dict"):
        if state_key in checkpoint and isinstance(checkpoint[state_key], Mapping):
            checkpoint = checkpoint[state_key]
            break

    state_dict = {
        key.removeprefix("module."): value for key, value in checkpoint.items()
    }
    generator = Generator().to(device)
    generator.load_state_dict(state_dict, strict=True)
    generator.eval()
    return generator


def load_serialized_file(path: Path) -> Any:
    """Load a supported embedding file.

    PyTorch ``.pt`` files can use pickle. Only load files from trusted sources.
    """

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    if suffix in {".pt", ".pth"}:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".csv":
        return np.loadtxt(path, delimiter=",", dtype=np.float32)
    if suffix == ".txt":
        return np.loadtxt(path, dtype=np.float32)

    raise ValueError(
        f"Unsupported file extension '{suffix}' for {path}. "
        "See --help for supported formats."
    )


def load_image_embeddings(path: Path) -> torch.Tensor:
    """Load one image embedding or a batch of image embeddings."""

    loaded = load_serialized_file(path)
    if isinstance(loaded, Mapping):
        loaded = _extract_first_present(
            loaded,
            (
                "image_embedding",
                "image_embeddings",
                "embedding",
                "embeddings",
                "test_img",
                "unseen_img",
                "image",
                "images",
            ),
            input_name="test image",
        )
    return to_embedding_matrix(loaded, input_name="test image")


def load_text_candidates(path: Path) -> tuple[list[str], torch.Tensor]:
    """Load labeled candidate text embeddings and deduplicate by label."""

    loaded = load_serialized_file(path)
    labels: Any
    embeddings: Any

    if isinstance(loaded, Mapping):
        key_pairs = (
            ("labels", "embeddings"),
            ("text_labels", "text_embeddings"),
            ("test_label", "test_text"),
            ("unseen_label", "unseen_text"),
        )
        for label_key, embedding_key in key_pairs:
            if label_key in loaded and embedding_key in loaded:
                labels = loaded[label_key]
                embeddings = loaded[embedding_key]
                break
        else:
            # A direct mapping such as {"cat": tensor(...), "dog": tensor(...)}.
            labels = list(loaded)
            embeddings = list(loaded.values())
    elif isinstance(loaded, Sequence) and not isinstance(loaded, (str, bytes)):
        if not loaded or not all(isinstance(item, Mapping) for item in loaded):
            raise ValueError(
                "A text-candidate list must contain objects with 'label' and "
                "'embedding' fields."
            )
        labels = [item["label"] for item in loaded]
        embeddings = [item["embedding"] for item in loaded]
    else:
        raise TypeError(
            "Text candidates must be a labeled mapping or a list of labeled "
            "embedding objects."
        )

    label_list = _to_label_list(labels)
    embedding_matrix = to_embedding_matrix(
        embeddings,
        input_name="candidate text",
    )
    if len(label_list) != len(embedding_matrix):
        raise ValueError(
            "Candidate labels and text embeddings must have equal lengths; "
            f"received {len(label_list)} and {len(embedding_matrix)}."
        )

    # Match the training/evaluation convention: for duplicate labels, retain
    # the final text embedding while preserving first-seen label order.
    text_by_label: dict[str, torch.Tensor] = {}
    for label, embedding in zip(label_list, embedding_matrix):
        text_by_label[label] = embedding

    unique_labels = list(text_by_label)
    unique_embeddings = torch.stack(
        [text_by_label[label] for label in unique_labels]
    )
    return unique_labels, unique_embeddings


def _extract_first_present(
    values: Mapping[str, Any],
    keys: Sequence[str],
    input_name: str,
) -> Any:
    for key in keys:
        if key in values:
            return values[key]
    raise KeyError(
        f"Could not find {input_name} embeddings. Expected one of: "
        + ", ".join(keys)
    )


def _to_label_list(labels: Any) -> list[str]:
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().tolist()
    elif isinstance(labels, np.ndarray):
        labels = labels.tolist()
    elif isinstance(labels, (str, bytes)):
        labels = [labels]

    try:
        return [str(label) for label in labels]
    except TypeError as exc:
        raise TypeError("Candidate labels must be an iterable collection.") from exc


def to_embedding_matrix(values: Any, input_name: str) -> torch.Tensor:
    """Convert common array containers to a float32 matrix with 512 columns."""

    if isinstance(values, (list, tuple)) and values:
        rows = [torch.as_tensor(value, dtype=torch.float32) for value in values]
        if all(row.ndim == 1 for row in rows):
            matrix = torch.stack(rows)
        else:
            matrix = torch.as_tensor(values, dtype=torch.float32)
    else:
        matrix = torch.as_tensor(values, dtype=torch.float32)

    if matrix.ndim == 1:
        matrix = matrix.unsqueeze(0)
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"{input_name.title()} embeddings have shape {tuple(matrix.shape)}; "
            f"expected ({EMBEDDING_DIM},) or (N, {EMBEDDING_DIM})."
        )
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{input_name.title()} embeddings contain NaN or infinity.")
    return matrix


def predict(
    generator: Generator,
    image_embeddings: torch.Tensor,
    candidate_labels: Sequence[str],
    candidate_text_embeddings: torch.Tensor,
    device: torch.device,
    top_k: int,
) -> list[dict[str, Any]]:
    """Generate visual prototypes and rank them by image dot product."""

    if not candidate_labels:
        raise ValueError("At least one candidate text label is required.")

    top_k = min(top_k, len(candidate_labels))
    with torch.no_grad():
        image_embeddings = image_embeddings.to(device)
        candidate_text_embeddings = candidate_text_embeddings.to(device)
        generated_prototypes = generator(candidate_text_embeddings)
        scores = image_embeddings @ generated_prototypes.t()
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)

    results: list[dict[str, Any]] = []
    for image_index in range(len(image_embeddings)):
        predictions = [
            {
                "rank": rank + 1,
                "label": candidate_labels[int(candidate_index)],
                "score": float(score),
            }
            for rank, (candidate_index, score) in enumerate(
                zip(top_indices[image_index].tolist(), top_scores[image_index].tolist())
            )
        ]
        results.append(
            {
                "image_index": image_index,
                "predicted_label": predictions[0]["label"],
                "predictions": predictions,
            }
        )
    return results


def print_results(results: Sequence[Mapping[str, Any]]) -> None:
    for result in results:
        print(
            f"Image {result['image_index']}: "
            f"{result['predicted_label']}"
        )
        for prediction in result["predictions"]:
            print(
                f"  {prediction['rank']}. {prediction['label']} "
                f"(score: {prediction['score']:.6f})"
            )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    generator = load_generator(args.checkpoint_path, device)
    image_embeddings = load_image_embeddings(args.test_image)
    candidate_labels, candidate_text_embeddings = load_text_candidates(
        args.test_text
    )

    results = predict(
        generator=generator,
        image_embeddings=image_embeddings,
        candidate_labels=candidate_labels,
        candidate_text_embeddings=candidate_text_embeddings,
        device=device,
        top_k=args.top_k,
    )
    print_results(results)

    if args.json_output is not None:
        json_output = args.json_output.expanduser().resolve()
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
        print(f"JSON results: {json_output}")


if __name__ == "__main__":
    main()
