#!/usr/bin/env python3
"""Train AAG, a prototype-based WGAN-style model for zero-shot learning.

The public interface intentionally accepts one complete preprocessed dataset
file. Dataset split selection belongs in preprocessing, not in this trainer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


EMBEDDING_DIM = 512
REQUIRED_DATASET_KEYS = (
    "seen_img",
    "seen_text",
    "seen_label",
    "unseen_img",
    "unseen_text",
    "unseen_label",
)


@dataclass
class TrainingHistory:
    """Metrics collected while training."""

    discriminator_losses: list[float]
    generator_losses: list[float]
    zsl_accuracies: list[float]
    best_accuracy: float
    best_epoch: int


class CustomDataset(Dataset):
    """Pair class-level visual prototypes with their text embeddings."""

    def __init__(self, clip_image: Sequence[Any], clip_text: Sequence[Any]) -> None:
        if len(clip_image) != len(clip_text):
            raise ValueError(
                "Visual prototypes and text embeddings must have equal lengths; "
                f"received {len(clip_image)} and {len(clip_text)}."
            )

        self.clip_image = [
            torch.as_tensor(image, dtype=torch.float32) for image in clip_image
        ]
        self.clip_text = [
            torch.as_tensor(text, dtype=torch.float32) for text in clip_text
        ]

    def __len__(self) -> int:
        return len(self.clip_text)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.clip_image[idx], self.clip_text[idx]


class Generator(nn.Module):
    """Map a 512-dimensional text embedding to a visual prototype."""

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


class Discriminator(nn.Module):
    """Score real and generated visual prototypes."""

    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(EMBEDDING_DIM, 256)
        self.output = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.output(self.relu(self.input(input_tensor)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AAG on one complete preprocessed .pt dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Full path to the preprocessed .pt dataset file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./checkpoint"),
        help="Directory for the best checkpoint and training log.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--n-critic",
        type=int,
        default=1,
        help="Number of critic steps between generator updates.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Training device. 'auto' uses CUDA when available, otherwise CPU.",
    )

    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.lr <= 0:
        parser.error("--lr must be greater than zero")
    if args.n_critic <= 0:
        parser.error("--n-critic must be greater than zero")
    return args


def resolve_device(device_arg: str) -> torch.device:
    """Resolve a CLI-friendly device setting to a PyTorch device."""

    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def load_dataset(dataset_path: Path) -> dict[str, Any]:
    """Load and validate a trusted preprocessed dataset.

    ``weights_only=False`` is required for common AAG datasets containing NumPy
    arrays. PyTorch/pickle files can execute code while loading, so callers
    should only use dataset files from a trusted source.
    """

    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    try:
        loaded_dataset = torch.load(
            dataset_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # Compatibility with PyTorch versions predating ``weights_only``.
        loaded_dataset = torch.load(dataset_path, map_location="cpu")

    if not isinstance(loaded_dataset, dict):
        raise TypeError("Expected the dataset file to contain a dictionary.")

    validate_dataset(loaded_dataset)
    return loaded_dataset


def validate_dataset(dataset: Mapping[str, Any]) -> None:
    """Validate the fields and embedding shapes used by the training loop."""

    missing_keys = [key for key in REQUIRED_DATASET_KEYS if key not in dataset]
    if missing_keys:
        raise KeyError(
            "Dataset is missing required key(s): " + ", ".join(missing_keys)
        )

    for partition in ("seen", "unseen"):
        images = dataset[f"{partition}_img"]
        texts = dataset[f"{partition}_text"]
        labels = dataset[f"{partition}_label"]

        lengths = (len(images), len(texts), len(labels))
        if len(set(lengths)) != 1:
            raise ValueError(
                f"{partition.title()} image, text, and label collections must "
                f"have equal lengths; received {lengths}."
            )
        if len(labels) == 0:
            raise ValueError(f"The {partition} partition must not be empty.")

        _validate_embeddings(f"{partition}_img", images)
        _validate_embeddings(f"{partition}_text", texts)
        _validate_labels(f"{partition}_label", labels)


def _validate_embeddings(name: str, embeddings: Sequence[Any]) -> None:
    for index, embedding in enumerate(embeddings):
        shape = tuple(torch.as_tensor(embedding).shape)
        if shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"{name}[{index}] has shape {shape}; expected "
                f"({EMBEDDING_DIM},)."
            )


def _validate_labels(name: str, labels: Sequence[Any]) -> None:
    for index, label in enumerate(labels):
        try:
            hash(label)
        except TypeError as exc:
            raise TypeError(
                f"{name}[{index}] must be a hashable scalar label."
            ) from exc


def group_images_by_label(
    images: Sequence[Any],
    labels: Sequence[Hashable],
) -> dict[Hashable, list[Any]]:
    """Group images in first-seen label order."""

    grouped_images: dict[Hashable, list[Any]] = {}
    for image, label in zip(images, labels):
        grouped_images.setdefault(label, []).append(image)
    return grouped_images


def prepare_training_data(
    dataset: Mapping[str, Any],
) -> tuple[list[torch.Tensor], list[Any], list[Hashable]]:
    """Create one visual prototype and one semantic embedding per seen class."""

    seen_img = dataset["seen_img"]
    seen_text = dataset["seen_text"]
    seen_label = dataset["seen_label"]

    grouped_images = group_images_by_label(seen_img, seen_label)

    # This preserves the notebook's dict(zip(...)) behavior: if a class appears
    # more than once, its last text embedding is paired with the prototype.
    text_by_label = dict(zip(seen_label, seen_text))

    train_labels = list(grouped_images)
    train_images = [
        torch.stack(
            [
                torch.as_tensor(image, dtype=torch.float32)
                for image in grouped_images[label]
            ]
        ).mean(dim=0)
        for label in train_labels
    ]
    train_text = [text_by_label[label] for label in train_labels]
    return train_images, train_text, train_labels


def build_dataloader(
    train_img: Sequence[Any],
    train_text: Sequence[Any],
    batch_size: int,
    seed: int = 42,
) -> DataLoader:
    data = CustomDataset(train_img, train_text)
    shuffle_generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )


def evaluate_with_dataset(
    generator: nn.Module,
    dataset: Mapping[str, Any],
    device: torch.device,
) -> float:
    """Compute top-1 zero-shot accuracy on all unseen examples."""

    unseen_img = dataset["unseen_img"]
    unseen_text = dataset["unseen_text"]
    labels = dataset["unseen_label"]

    # As in the original notebook, the last text embedding observed for each
    # unseen label becomes that class's semantic prototype.
    text_by_label = dict(zip(labels, unseen_text))
    class_labels = list(text_by_label)
    class_text = torch.stack(
        [
            torch.as_tensor(text_by_label[label], dtype=torch.float32)
            for label in class_labels
        ]
    ).to(device)

    was_training = generator.training
    generator.eval()
    correct = 0

    with torch.no_grad():
        generated_prototypes = generator(class_text)
        for test_img, target_label in zip(unseen_img, labels):
            image_embedding = torch.as_tensor(
                test_img,
                dtype=torch.float32,
                device=device,
            )
            logits = image_embedding @ generated_prototypes.t()
            prediction_index = int(torch.argmax(logits).item())
            if class_labels[prediction_index] == target_label:
                correct += 1

    if was_training:
        generator.train()
    return correct / len(unseen_img)


def train_generator(
    train_loader: DataLoader,
    dataset: Mapping[str, Any],
    device: torch.device,
    epochs: int,
    lr: float,
    n_critic: int,
    seed: int,
    output_dir: Path,
    dataset_name: str,
) -> TrainingHistory:
    """Train the original AAG objective and save the best generator."""

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=lr)
    optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=lr)

    discriminator_losses: list[float] = []
    generator_losses: list[float] = []
    zsl_accuracies: list[float] = []
    best_accuracy = -1.0
    best_epoch = 0

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"wgan_prototype_{dataset_name}.pth"

    for epoch_index in range(epochs):
        discriminator.train()
        generator.train()

        epoch_loss_d = 0.0
        epoch_loss_g = 0.0
        critic_updates = 0
        generator_updates = 0

        for batch_index, (img, text) in enumerate(train_loader):
            img = img.to(device=device, dtype=torch.float32)
            text = text.to(device=device, dtype=torch.float32)

            # Update the discriminator/critic using the notebook's objective.
            optimizer_d.zero_grad(set_to_none=True)
            real_output = discriminator(img)
            fake_images = generator(text)
            fake_output = discriminator(fake_images.detach())
            loss_d = torch.mean(real_output) - torch.mean(fake_output)
            loss_d.backward()
            optimizer_d.step()

            with torch.no_grad():
                for parameter in discriminator.parameters():
                    parameter.clamp_(-0.01, 0.01)

            epoch_loss_d += loss_d.item()
            critic_updates += 1

            if (batch_index + 1) % n_critic == 0:
                optimizer_g.zero_grad(set_to_none=True)
                generated_images = generator(text)
                loss_g = torch.mean(discriminator(generated_images))
                loss_g.backward()
                optimizer_g.step()

                epoch_loss_g += loss_g.item()
                generator_updates += 1

        if generator_updates == 0:
            raise ValueError(
                "--n-critic is larger than the number of batches, so the "
                "generator was never updated."
            )

        average_loss_d = epoch_loss_d / critic_updates
        average_loss_g = epoch_loss_g / generator_updates
        discriminator_losses.append(average_loss_d)
        generator_losses.append(average_loss_g)

        zsl_accuracy = evaluate_with_dataset(generator, dataset, device)
        zsl_accuracies.append(zsl_accuracy)

        if zsl_accuracy > best_accuracy:
            best_accuracy = zsl_accuracy
            best_epoch = epoch_index + 1
            torch.save(generator.state_dict(), checkpoint_path)

        print(
            f"Epoch [{epoch_index + 1}/{epochs}] "
            f"Loss_D: {average_loss_d:.6f} "
            f"Loss_G: {average_loss_g:.6f} "
            f"ZSL_accuracy: {zsl_accuracy:.6f}",
            end="\r",
            flush=True,
        )

    print()
    history = TrainingHistory(
        discriminator_losses=discriminator_losses,
        generator_losses=generator_losses,
        zsl_accuracies=zsl_accuracies,
        best_accuracy=best_accuracy,
        best_epoch=best_epoch,
    )
    log_path = output_dir / f"LOG-{dataset_name}.txt"
    log_path.write_text(
        f"Best acc: {history.best_accuracy:.4f}\n"
        f"Epoch: {history.best_epoch}\n",
        encoding="utf-8",
    )
    return history


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    device = resolve_device(args.device)

    dataset = load_dataset(dataset_path)
    train_img, train_text, train_labels = prepare_training_data(dataset)
    train_loader = build_dataloader(
        train_img,
        train_text,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    dataset_name = dataset_path.stem
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")
    print(f"Seen class prototypes: {len(train_labels)}")

    history = train_generator(
        train_loader=train_loader,
        dataset=dataset,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        n_critic=args.n_critic,
        seed=args.seed,
        output_dir=output_dir,
        dataset_name=dataset_name,
    )

    checkpoint_path = output_dir / f"wgan_prototype_{dataset_name}.pth"
    log_path = output_dir / f"LOG-{dataset_name}.txt"

    print(f"Best accuracy: {history.best_accuracy:.4f}")
    print(f"Best epoch: {history.best_epoch}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
