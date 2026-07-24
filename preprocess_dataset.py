#!/usr/bin/env python3
"""Prepare folder-based emotion datasets for CLIP fine-tuning and AAG.

Workflow:

1. ``manifest`` scans ``dataset_root/emotion_name/images...`` and writes the
   ``.npz`` consumed by ``finetune_clip.py``.
2. ``embeddings`` loads the fine-tuned CLIP checkpoint and writes the six-key
   ``.pt`` dataset consumed by ``AAG_training.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DEFAULT_IMAGE_EXTENSIONS = (
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
)
DEFAULT_PROMPT_TEMPLATE = (
    "A picture seems to express some feelings like {label}."
)
AAG_EMBEDDING_DIM = 512


class ImagePathDataset(Dataset):
    """Load and preprocess images while preserving manifest order."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        preprocess: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.image_paths = list(image_paths)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        try:
            with Image.open(image_path) as image:
                return self.preprocess(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Could not load image: {image_path}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a folder-based image dataset for CLIP fine-tuning and AAG."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Scan emotion folders and create the CLIP fine-tuning manifest.",
    )
    manifest_parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root containing one direct subdirectory per emotion class.",
    )
    manifest_parser.add_argument(
        "--output-path",
        type=Path,
        help=(
            "Manifest destination. Defaults to "
            "./data/<dataset-root-name>_manifest.npz."
        ),
    )
    manifest_parser.add_argument(
        "--unseen-classes",
        nargs="+",
        required=True,
        help=(
            "Paper-defined emotion classes reserved for zero-shot evaluation. "
            "All other class folders become seen."
        ),
    )
    manifest_parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_IMAGE_EXTENSIONS),
        help="Image extensions to include.",
    )
    manifest_parser.add_argument(
        "--skip-image-verification",
        action="store_true",
        help="Skip PIL verification of every discovered image.",
    )
    manifest_parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt containing a {label} placeholder.",
    )

    embeddings_parser = subparsers.add_parser(
        "embeddings",
        help="Encode the manifest with fine-tuned CLIP for AAG.",
    )
    embeddings_parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to the .npz produced by the manifest command.",
    )
    embeddings_parser.add_argument(
        "--clip-checkpoint",
        type=Path,
        required=True,
        help="Path to the checkpoint produced by finetune_clip.py.",
    )
    embeddings_parser.add_argument(
        "--output-path",
        type=Path,
        help=(
            "AAG dataset destination. Defaults to "
            "./data/<manifest-name>_aag.pt."
        ),
    )
    embeddings_parser.add_argument("--batch-size", type=int, default=64)
    embeddings_parser.add_argument("--num-workers", type=int, default=0)
    embeddings_parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    embeddings_parser.add_argument(
        "--download-root",
        type=Path,
        help="Optional directory for downloaded OpenAI CLIP weights.",
    )
    embeddings_parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Keep raw CLIP features instead of L2-normalizing them.",
    )
    return parser


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.command == "manifest":
        if "{label}" not in args.prompt_template:
            parser.error("--prompt-template must contain {label}")
    elif args.command == "embeddings":
        if args.batch_size <= 0:
            parser.error("--batch-size must be greater than zero")
        if args.num_workers < 0:
            parser.error("--num-workers must not be negative")


def normalize_extensions(extensions: Sequence[str]) -> set[str]:
    normalized = {
        extension.lower()
        if extension.startswith(".")
        else f".{extension.lower()}"
        for extension in extensions
    }
    if not normalized:
        raise ValueError("At least one image extension is required.")
    return normalized


def scan_class_folders(
    dataset_root: Path,
    extensions: set[str],
    verify_images: bool,
) -> dict[str, list[Path]]:
    """Discover images below each direct child directory."""

    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root not found: {dataset_root}")

    class_directories = sorted(
        (
            path
            for path in dataset_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=lambda path: path.name.casefold(),
    )
    if len(class_directories) < 2:
        raise ValueError(
            "The dataset root must contain at least two emotion folders."
        )

    images_by_label: dict[str, list[Path]] = {}
    source_names: dict[str, str] = {}
    invalid_images: list[Path] = []

    for class_directory in class_directories:
        label = class_directory.name.strip().lower()
        if not label:
            raise ValueError(f"Empty class folder name: {class_directory}")
        if label in images_by_label:
            raise ValueError(
                "Class folder names must be unique when lowercased: "
                f"'{source_names[label]}' and '{class_directory.name}'."
            )

        image_paths = sorted(
            (
                path.resolve()
                for path in class_directory.rglob("*")
                if path.is_file() and path.suffix.lower() in extensions
            ),
            key=lambda path: str(path).casefold(),
        )
        if not image_paths:
            raise ValueError(
                f"No supported images found in class folder: {class_directory}"
            )

        if verify_images:
            for image_path in image_paths:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except Exception:
                    invalid_images.append(image_path)

        images_by_label[label] = image_paths
        source_names[label] = class_directory.name

    if invalid_images:
        preview = "\n".join(f"  - {path}" for path in invalid_images[:5])
        remainder = len(invalid_images) - 5
        if remainder > 0:
            preview += f"\n  ... and {remainder} more"
        raise ValueError(
            f"{len(invalid_images)} unreadable image(s) found:\n{preview}"
        )
    return images_by_label


def split_classes(
    class_labels: Sequence[str],
    unseen_classes: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Apply the paper-defined, disjoint seen and unseen class split."""

    available = sorted(set(class_labels))
    if len(available) < 2:
        raise ValueError("At least two classes are required.")

    requested = [label.strip().lower() for label in unseen_classes]
    if len(set(requested)) != len(requested):
        raise ValueError("--unseen-classes contains duplicate labels.")
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError("Unknown unseen class(es): " + ", ".join(unknown))
    unseen = sorted(requested)

    seen = sorted(set(available) - set(unseen))
    if not seen or not unseen:
        raise ValueError("Both seen and unseen class partitions must be non-empty.")
    if len(seen) < 2:
        raise ValueError(
            "At least two seen classes are required for CLIP fine-tuning and AAG."
        )
    return seen, unseen


def flatten_partition(
    images_by_label: Mapping[str, Sequence[Path]],
    class_labels: Sequence[str],
) -> tuple[list[Path], list[str]]:
    image_paths: list[Path] = []
    labels: list[str] = []
    for label in class_labels:
        class_images = images_by_label[label]
        image_paths.extend(class_images)
        labels.extend([label] * len(class_images))
    return image_paths, labels


def save_manifest(
    output_path: Path,
    dataset_root: Path,
    seen_paths: Sequence[Path],
    seen_labels: Sequence[str],
    unseen_paths: Sequence[Path],
    unseen_labels: Sequence[str],
    seen_classes: Sequence[str],
    unseen_classes: Sequence[str],
    prompt_template: str,
) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        seen=np.asarray([str(path) for path in seen_paths]),
        seen_label=np.asarray(seen_labels),
        unseen=np.asarray([str(path) for path in unseen_paths]),
        unseen_label=np.asarray(unseen_labels),
        seen_classes=np.asarray(seen_classes),
        unseen_classes=np.asarray(unseen_classes),
        dataset_root=np.asarray(str(dataset_root.expanduser().resolve())),
        prompt_template=np.asarray(prompt_template),
    )
    return output_path


def create_manifest(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset_root.expanduser().resolve()
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else (
            Path("./data") / f"{dataset_root.name}_manifest.npz"
        ).resolve()
    )
    images_by_label = scan_class_folders(
        dataset_root=dataset_root,
        extensions=normalize_extensions(args.extensions),
        verify_images=not args.skip_image_verification,
    )
    seen_classes, unseen_classes = split_classes(
        class_labels=list(images_by_label),
        unseen_classes=args.unseen_classes,
    )
    seen_paths, seen_labels = flatten_partition(
        images_by_label,
        seen_classes,
    )
    unseen_paths, unseen_labels = flatten_partition(
        images_by_label,
        unseen_classes,
    )
    save_manifest(
        output_path=output_path,
        dataset_root=dataset_root,
        seen_paths=seen_paths,
        seen_labels=seen_labels,
        unseen_paths=unseen_paths,
        unseen_labels=unseen_labels,
        seen_classes=seen_classes,
        unseen_classes=unseen_classes,
        prompt_template=args.prompt_template,
    )

    print(f"Dataset root: {dataset_root}")
    print(
        f"Seen: {len(seen_classes)} classes, {len(seen_paths)} images "
        f"({', '.join(seen_classes)})"
    )
    print(
        f"Unseen: {len(unseen_classes)} classes, {len(unseen_paths)} images "
        f"({', '.join(unseen_classes)})"
    )
    print(f"Manifest: {output_path}")
    return output_path


def load_manifest(
    manifest_path: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if manifest_path.suffix.lower() != ".npz":
        raise ValueError("The manifest must be a .npz file.")

    with np.load(manifest_path, allow_pickle=False) as archive:
        required_keys = ("seen", "seen_label", "unseen", "unseen_label")
        missing = [key for key in required_keys if key not in archive]
        if missing:
            raise KeyError(
                "Manifest is missing required key(s): " + ", ".join(missing)
            )
        manifest = {key: archive[key].copy() for key in archive.files}

    for partition in ("seen", "unseen"):
        paths = manifest[partition]
        labels = manifest[f"{partition}_label"]
        if paths.ndim != 1 or labels.ndim != 1:
            raise ValueError(
                f"{partition} paths and labels must be one-dimensional."
            )
        if len(paths) != len(labels):
            raise ValueError(
                f"{partition} paths and labels have different lengths."
            )
        if len(paths) == 0:
            raise ValueError(f"The {partition} partition must not be empty.")

    return manifest


def manifest_paths_and_labels(
    manifest: Mapping[str, Any],
    partition: str,
) -> tuple[list[Path], list[str]]:
    paths = [Path(str(value)).expanduser().resolve() for value in manifest[partition]]
    labels = [str(value) for value in manifest[f"{partition}_label"]]

    missing_paths = [path for path in paths if not path.is_file()]
    if missing_paths:
        preview = "\n".join(f"  - {path}" for path in missing_paths[:5])
        remainder = len(missing_paths) - 5
        if remainder > 0:
            preview += f"\n  ... and {remainder} more"
        raise FileNotFoundError(
            f"{len(missing_paths)} {partition} image(s) are missing:\n{preview}"
        )
    return paths, labels


def import_openai_clip() -> Any:
    try:
        import clip
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI CLIP is not installed. Run "
            "'pip install -r requirements.txt' first."
        ) from exc
    return clip


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def load_clip_checkpoint(
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CLIP checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError("Expected the CLIP checkpoint to contain a dictionary.")
    required_keys = ("model_name", "model_state_dict")
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise KeyError(
            "CLIP checkpoint is missing required key(s): " + ", ".join(missing)
        )
    return checkpoint


def encode_images(
    model: torch.nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    image_paths: Sequence[Path],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    normalize: bool,
) -> torch.Tensor:
    dataset = ImagePathDataset(image_paths, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    batches: list[torch.Tensor] = []
    with torch.no_grad():
        for images in tqdm(dataloader, desc="Encoding images"):
            features = model.encode_image(
                images.to(device, non_blocking=device.type == "cuda")
            ).float()
            if normalize:
                features = F.normalize(features, dim=-1)
            batches.append(features.cpu())
    return torch.cat(batches)


def encode_text_by_label(
    model: torch.nn.Module,
    tokenize: Callable[..., torch.Tensor],
    labels: Sequence[str],
    prompt_template: str,
    batch_size: int,
    device: torch.device,
    normalize: bool,
) -> dict[str, torch.Tensor]:
    unique_labels = list(dict.fromkeys(labels))
    prompts = [prompt_template.format(label=label) for label in unique_labels]
    features_by_label: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for start in tqdm(
            range(0, len(prompts), batch_size),
            desc="Encoding text",
        ):
            batch_labels = unique_labels[start : start + batch_size]
            batch_prompts = prompts[start : start + batch_size]
            tokens = tokenize(batch_prompts, truncate=True).to(device)
            features = model.encode_text(tokens).float()
            if normalize:
                features = F.normalize(features, dim=-1)
            for label, feature in zip(batch_labels, features.cpu()):
                features_by_label[label] = feature
    return features_by_label


def assemble_aag_dataset(
    seen_image_embeddings: torch.Tensor,
    seen_labels: Sequence[str],
    unseen_image_embeddings: torch.Tensor,
    unseen_labels: Sequence[str],
    text_by_label: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    seen_text = torch.stack([text_by_label[label] for label in seen_labels])
    unseen_text = torch.stack([text_by_label[label] for label in unseen_labels])

    dataset = {
        "seen_img": seen_image_embeddings.float().cpu(),
        "seen_text": seen_text.float().cpu(),
        "seen_label": list(seen_labels),
        "unseen_img": unseen_image_embeddings.float().cpu(),
        "unseen_text": unseen_text.float().cpu(),
        "unseen_label": list(unseen_labels),
        "metadata": dict(metadata),
    }
    validate_aag_shapes(dataset)
    return dataset


def validate_aag_shapes(dataset: Mapping[str, Any]) -> None:
    for partition in ("seen", "unseen"):
        image_embeddings = dataset[f"{partition}_img"]
        text_embeddings = dataset[f"{partition}_text"]
        labels = dataset[f"{partition}_label"]
        expected_shape = (len(labels), AAG_EMBEDDING_DIM)

        if tuple(image_embeddings.shape) != expected_shape:
            raise ValueError(
                f"{partition}_img has shape {tuple(image_embeddings.shape)}; "
                f"AAG requires {expected_shape}."
            )
        if tuple(text_embeddings.shape) != expected_shape:
            raise ValueError(
                f"{partition}_text has shape {tuple(text_embeddings.shape)}; "
                f"AAG requires {expected_shape}."
            )
        if not torch.isfinite(image_embeddings).all():
            raise ValueError(f"{partition}_img contains NaN or infinity.")
        if not torch.isfinite(text_embeddings).all():
            raise ValueError(f"{partition}_text contains NaN or infinity.")


def create_aag_embeddings(args: argparse.Namespace) -> Path:
    manifest_path = args.manifest_path.expanduser().resolve()
    checkpoint_path = args.clip_checkpoint.expanduser().resolve()
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else (
            Path("./data") / f"{manifest_path.stem}_aag.pt"
        ).resolve()
    )
    device = resolve_device(args.device)
    normalize = not args.no_normalize

    manifest = load_manifest(manifest_path)
    seen_paths, seen_labels = manifest_paths_and_labels(manifest, "seen")
    unseen_paths, unseen_labels = manifest_paths_and_labels(manifest, "unseen")
    checkpoint = load_clip_checkpoint(checkpoint_path)

    model_name = str(checkpoint["model_name"])
    prompt_template = str(
        checkpoint.get(
            "prompt_template",
            manifest.get("prompt_template", DEFAULT_PROMPT_TEMPLATE),
        )
    )
    clip_module = import_openai_clip()
    if model_name not in clip_module.available_models():
        raise ValueError(f"Unknown OpenAI CLIP model in checkpoint: {model_name}")

    load_options: dict[str, Any] = {}
    if args.download_root is not None:
        load_options["download_root"] = str(
            args.download_root.expanduser().resolve()
        )
    model, preprocess = clip_module.load(
        model_name,
        device=device,
        jit=False,
        **load_options,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    print(f"Manifest: {manifest_path}")
    print(f"CLIP checkpoint: {checkpoint_path}")
    print(f"CLIP model: {model_name}")
    print(f"Device: {device}")
    print(f"Normalize embeddings: {normalize}")

    seen_image_embeddings = encode_images(
        model=model,
        preprocess=preprocess,
        image_paths=seen_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        normalize=normalize,
    )
    unseen_image_embeddings = encode_images(
        model=model,
        preprocess=preprocess,
        image_paths=unseen_paths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        normalize=normalize,
    )
    text_by_label = encode_text_by_label(
        model=model,
        tokenize=clip_module.tokenize,
        labels=seen_labels + unseen_labels,
        prompt_template=prompt_template,
        batch_size=args.batch_size,
        device=device,
        normalize=normalize,
    )
    dataset = assemble_aag_dataset(
        seen_image_embeddings=seen_image_embeddings,
        seen_labels=seen_labels,
        unseen_image_embeddings=unseen_image_embeddings,
        unseen_labels=unseen_labels,
        text_by_label=text_by_label,
        metadata={
            "source_manifest": str(manifest_path),
            "clip_checkpoint": str(checkpoint_path),
            "clip_model": model_name,
            "prompt_template": prompt_template,
            "normalized": normalize,
            "embedding_dim": AAG_EMBEDDING_DIM,
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)
    print(
        f"AAG dataset: {output_path} "
        f"({len(seen_labels)} seen images, "
        f"{len(unseen_labels)} unseen images)"
    )
    return output_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    if args.command == "manifest":
        create_manifest(args)
    elif args.command == "embeddings":
        create_aag_embeddings(args)


if __name__ == "__main__":
    main()
