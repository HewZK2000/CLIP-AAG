#!/usr/bin/env python3
"""Fine-tune OpenAI CLIP on image paths stored in a NumPy dataset.

The input ``.npz`` file must contain a ``seen`` array of image paths. Each
image's class label is inferred from its parent directory and inserted into the
configurable prompt template.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DEFAULT_PROMPT_TEMPLATE = (
    "A picture seems to express some feelings like {label}."
)


class ImageTextDataset(Dataset):
    """Return preprocessed images and tokenized text prompts."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        prompts: Sequence[str],
        preprocess: Callable[[Image.Image], torch.Tensor],
        tokenize: Callable[..., torch.Tensor],
    ) -> None:
        if len(image_paths) != len(prompts):
            raise ValueError(
                "Image paths and prompts must have equal lengths; received "
                f"{len(image_paths)} and {len(prompts)}."
            )

        self.image_paths = list(image_paths)
        self.preprocess = preprocess
        self.tokenized_prompts = tokenize(list(prompts), truncate=True)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        try:
            with Image.open(image_path) as image:
                processed_image = self.preprocess(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Could not load image: {image_path}") from exc

        return processed_image, self.tokenized_prompts[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune OpenAI CLIP using the image paths under the 'seen' key "
            "of one complete .npz dataset."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Full path to the .npz file containing a 'seen' image-path array.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help=(
            "Checkpoint destination. Defaults to "
            "./models/clip_finetuned_<dataset-name>.pt."
        ),
    )
    parser.add_argument(
        "--model-name",
        default="ViT-B/16",
        help="OpenAI CLIP model name; defaults to ViT-B/16.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt containing a {label} placeholder.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes; defaults to 0 for portability.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Training device. 'auto' uses CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        help="Optional directory for downloaded OpenAI CLIP model weights.",
    )

    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero")
    if args.batch_size <= 1:
        parser.error("--batch-size must be greater than one for contrastive loss")
    if args.lr <= 0:
        parser.error("--lr must be greater than zero")
    if args.weight_decay < 0:
        parser.error("--weight-decay must not be negative")
    if args.num_workers < 0:
        parser.error("--num-workers must not be negative")
    if "{label}" not in args.prompt_template:
        parser.error("--prompt-template must contain the {label} placeholder")
    return args


def import_openai_clip() -> Any:
    """Import OpenAI CLIP with an actionable dependency error."""

    try:
        import clip
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI CLIP is not installed. Run "
            "'pip install -r requirements.txt' before training."
        ) from exc
    return clip


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_samples(
    dataset_path: Path,
    prompt_template: str,
) -> tuple[list[Path], list[str]]:
    """Load seen image paths and derive prompts from parent-folder labels."""

    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if dataset_path.suffix.lower() != ".npz":
        raise ValueError("The fine-tuning dataset must be a .npz file.")

    with np.load(dataset_path, allow_pickle=False) as dataset:
        if "seen" not in dataset:
            raise KeyError("Dataset is missing the required 'seen' array.")
        raw_image_paths = dataset["seen"]
        raw_labels = dataset["seen_label"] if "seen_label" in dataset else None

    if raw_image_paths.ndim != 1:
        raise ValueError(
            "The 'seen' array must be one-dimensional; received shape "
            f"{raw_image_paths.shape}."
        )
    if len(raw_image_paths) < 2:
        raise ValueError(
            "At least two seen images are required for contrastive fine-tuning."
        )
    if raw_labels is not None:
        if raw_labels.ndim != 1:
            raise ValueError("The optional 'seen_label' array must be 1D.")
        if len(raw_labels) != len(raw_image_paths):
            raise ValueError(
                "'seen' and 'seen_label' must have equal lengths."
            )

    image_paths: list[Path] = []
    prompts: list[str] = []
    missing_paths: list[Path] = []

    for index, raw_path in enumerate(raw_image_paths):
        if isinstance(raw_path, (bytes, np.bytes_)):
            path_value = raw_path.decode("utf-8")
        else:
            path_value = str(raw_path)
        image_path = Path(path_value).expanduser()
        if not image_path.is_absolute():
            image_path = dataset_path.parent / image_path
        image_path = image_path.resolve()

        if not image_path.is_file():
            missing_paths.append(image_path)

        label = (
            str(raw_labels[index]).strip().lower()
            if raw_labels is not None
            else image_path.parent.name.lower()
        )
        if not label:
            raise ValueError(
                f"Could not infer a parent-folder label from: {image_path}"
            )

        image_paths.append(image_path)
        prompts.append(prompt_template.format(label=label))

    if missing_paths:
        preview = "\n".join(f"  - {path}" for path in missing_paths[:5])
        remainder = len(missing_paths) - 5
        if remainder > 0:
            preview += f"\n  ... and {remainder} more"
        raise FileNotFoundError(
            f"{len(missing_paths)} image path(s) do not exist:\n{preview}"
        )

    return image_paths, prompts


def build_dataloader(
    image_paths: Sequence[Path],
    prompts: Sequence[str],
    preprocess: Callable[[Image.Image], torch.Tensor],
    tokenize: Callable[..., torch.Tensor],
    batch_size: int,
    num_workers: int,
    seed: int,
    use_cuda: bool,
) -> DataLoader:
    dataset = ImageTextDataset(
        image_paths=image_paths,
        prompts=prompts,
        preprocess=preprocess,
        tokenize=tokenize,
    )
    shuffle_generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        generator=shuffle_generator,
    )


def convert_model_and_gradients_to_fp32(model: nn.Module) -> None:
    """Convert CUDA CLIP weights and available gradients before AdamW.step."""

    for parameter in model.parameters():
        parameter.data = parameter.data.float()
        if parameter.grad is not None:
            parameter.grad.data = parameter.grad.data.float()


def train_one_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_module: Any,
    epoch_number: int,
    total_epochs: int,
) -> float:
    model.train()
    image_loss = nn.CrossEntropyLoss()
    text_loss = nn.CrossEntropyLoss()

    cumulative_loss = 0.0
    processed_examples = 0
    progress = tqdm(
        train_dataloader,
        total=len(train_dataloader),
        desc=f"Epoch {epoch_number}/{total_epochs}",
    )

    for images, texts in progress:
        images = images.to(device, non_blocking=device.type == "cuda")
        texts = texts.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)

        logits_per_image, logits_per_text = model(images, texts)
        ground_truth = torch.arange(
            len(images),
            dtype=torch.long,
            device=device,
        )
        total_loss = (
            image_loss(logits_per_image, ground_truth)
            + text_loss(logits_per_text, ground_truth)
        ) / 2

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite loss detected in epoch {epoch_number}."
            )

        total_loss.backward()
        if device.type == "cuda":
            convert_model_and_gradients_to_fp32(model)
            optimizer.step()
            clip_module.model.convert_weights(model)
        else:
            optimizer.step()

        batch_size = len(images)
        cumulative_loss += total_loss.item() * batch_size
        processed_examples += batch_size
        progress.set_postfix(loss=f"{total_loss.item():.4f}")

    return cumulative_loss / processed_examples


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    epoch_losses: Sequence[float],
    model_name: str,
    prompt_template: str,
) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_name": model_name,
        "prompt_template": prompt_template,
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": epoch_losses[-1],
        "epoch_losses": list(epoch_losses),
    }
    torch.save(checkpoint, output_path)


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else (
            Path("./models")
            / f"clip_finetuned_{dataset_path.stem}.pt"
        ).resolve()
    )
    device = resolve_device(args.device)
    set_random_seeds(args.seed)

    clip_module = import_openai_clip()
    available_models = clip_module.available_models()
    if args.model_name not in available_models:
        raise ValueError(
            f"Unknown CLIP model '{args.model_name}'. Available models: "
            + ", ".join(available_models)
        )

    image_paths, prompts = load_training_samples(
        dataset_path=dataset_path,
        prompt_template=args.prompt_template,
    )

    load_options: dict[str, Any] = {}
    if args.download_root is not None:
        load_options["download_root"] = str(
            args.download_root.expanduser().resolve()
        )
    model, preprocess = clip_module.load(
        args.model_name,
        device=device,
        jit=False,
        **load_options,
    )
    train_dataloader = build_dataloader(
        image_paths=image_paths,
        prompts=prompts,
        preprocess=preprocess,
        tokenize=clip_module.tokenize,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_cuda=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"Dataset: {dataset_path}")
    print(f"Training images: {len(image_paths)}")
    print(f"CLIP model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Output: {output_path}")

    epoch_losses: list[float] = []
    for epoch_number in range(1, args.epochs + 1):
        average_loss = train_one_epoch(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            clip_module=clip_module,
            epoch_number=epoch_number,
            total_epochs=args.epochs,
        )
        epoch_losses.append(average_loss)
        print(
            f"Epoch [{epoch_number}/{args.epochs}] "
            f"average loss: {average_loss:.6f}"
        )

    save_checkpoint(
        output_path=output_path,
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        epoch_losses=epoch_losses,
        model_name=args.model_name,
        prompt_template=args.prompt_template,
    )
    print(f"Checkpoint saved to: {output_path}")


if __name__ == "__main__":
    main()
