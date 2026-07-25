# CLIP-AAG: Zero-Shot Image Emotion Recognition

This repository contains the release implementation of the
CLIP-Affective Attribute Generator (CLIP-AAG) proposed in:

> **Beyond Illusions of Competence: Revisiting Zero-Shot Learning in Emotion
> Recognition with FEAr Dataset**  
> Zhong Ken Hew, Lai-Kuan Wong, and Chee Seng Chan

The paper studies three problems in zero-shot image emotion recognition:
performance variation caused by inconsistent seen-unseen splits,
non-standardized evaluation, and semantic attributes that align poorly with
emotional visual content.

It introduces:

- **FEAr (Fair Evaluation for Affective Recognition):** a dataset-agnostic
  framework for standardized, valence-balanced seen-unseen configurations.
- **FaH (FEAr-accuracy Harmonization):** a harmonic aggregation of accuracy
  across FEAr split configurations that emphasizes weak-performing splits.
- **CLIP-AAG:** a two-stage model that fine-tunes CLIP on seen emotion classes
  and uses a WGAN-style generator to enrich text attributes with affective
  visual information.

FEAr is an evaluation framework applied to existing datasets; this repository
does not redistribute FI, Emotion6, or ArtPhoto.

## Files

- `AAG_training.py` — canonical command-line training implementation
- `AAG_evaluate.py` — standalone checkpoint loading and prediction
- `finetune_clip.py` — OpenAI CLIP fine-tuning command-line program
- `preprocess_dataset.py` — raw-folder manifests and CLIP embedding extraction

Preprocessing, training, and prediction are provided as plain Python
command-line programs, so no notebook is required.

## Paper-to-code map

| Paper component | Release implementation |
| --- | --- |
| Explicit seen-unseen class configuration | `preprocess_dataset.py manifest` |
| Stage 1: task-specific CLIP fine-tuning | `finetune_clip.py` |
| Frozen CLIP image and text feature extraction | `preprocess_dataset.py embeddings` |
| Stage 2: Affective Attribute Generator | `AAG_training.py` |
| Unseen-class inference | `AAG_evaluate.py` |

The current CLI executes one explicitly declared FEAr split at a time. Use the
class combinations reported in Table 1 of the paper and run each configuration
independently when reproducing the full FaH evaluation.

## Installation

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

## FEAr evaluation protocol

FEAr requires disjoint seen and unseen emotion classes and balances emotional
valence in the unseen partition.

- **Ratio Configuration-I (Minor Split):** two unseen classes. When feasible,
  one positive-valence and one negative-valence emotion are selected.
- **Ratio Configuration-II (Major Split):** `floor(N / 2)` unseen classes,
  allocated to approximate the positive-negative valence distribution of the
  complete label space.

The paper instantiates the protocol as follows:

| Dataset | Classes | Configuration-I | Configuration-II | Valid splits |
| --- | ---: | --- | --- | ---: |
| FI | 8 | 6 seen / 2 unseen | 4 seen / 4 unseen | 16 + 36 |
| ArtPhoto | 8 | 6 seen / 2 unseen | 4 seen / 4 unseen | 16 + 36 |
| Emotion6 | 6 | 4 seen / 2 unseen | 3 seen / 3 unseen | 8 + 16 |

Together, these form 128 standardized configurations. The paper defines FaH
over the accuracy scores from the relevant configurations as:

```text
FaH = S / sum(1 / (accuracy_i + 0.1))
```

when at least one accuracy is non-zero; FaH is zero when all scores are zero.
This repository currently records per-split accuracy. Aggregate the intended
paper configurations when reporting FaH.

## Benchmarks

The paper evaluates the framework on three established image-emotion datasets:

| Dataset | Images used in the paper | Emotion taxonomy | Dataset access |
| --- | ---: | --- | --- |
| FI | 22,700 | Mikels' eight emotion categories | Obtain from the original dataset authors |
| ArtPhoto | 806 | Mikels' eight emotion categories | [ImageEmotion dataset page](https://www.imageemotion.org/) |
| Emotion6 | 1,980 | Ekman's six basic emotions | [Cornell Chen Lab downloads](http://chenlab.ece.cornell.edu/downloads.html) |

Obtain each dataset from its original source, comply with its license and usage
terms, and cite the corresponding dataset paper. Arrange downloaded images
under one emotion folder per class before running the preprocessing workflow.

## End-to-end dataset workflow

Start with one direct subfolder per emotion. Images below each emotion folder
are discovered recursively:

```text
emotion_dataset/
├── angry/
│   ├── image_001.jpg
│   └── image_002.jpg
├── happy/
│   ├── image_003.jpg
│   └── image_004.jpg
├── sad/
│   └── image_005.jpg
└── surprise/
    └── image_006.jpg
```

### 1. Create the seen/unseen manifest

For a publication split, explicitly name the emotion classes held out for
zero-shot evaluation:

```bash
python preprocess_dataset.py manifest \
  --dataset-root "/absolute/path/to/emotion_dataset" \
  --output-path "./data/emotions_manifest.npz" \
  --unseen-classes sad surprise
```

All remaining emotion folders become seen classes. The manifest stores image
paths, per-image labels, the explicit class partitions, and the prompt
template. Only seen images are used for CLIP fine-tuning; the paper-defined
unseen classes remain held out.

### 2. Fine-tune CLIP on seen classes

```bash
python finetune_clip.py \
  --dataset-path "./data/emotions_manifest.npz" \
  --output-path "./models/clip_finetuned_emotions.pt" \
  --epochs 30 \
  --batch-size 32 \
  --device auto
```

### 3. Encode images and class prompts for AAG

```bash
python preprocess_dataset.py embeddings \
  --manifest-path "./data/emotions_manifest.npz" \
  --clip-checkpoint "./models/clip_finetuned_emotions.pt" \
  --output-path "./data/emotions_aag.pt" \
  --batch-size 64 \
  --device auto
```

This sends every raw image through the fine-tuned CLIP image encoder and each
emotion prompt through its text encoder. By default, both outputs are
L2-normalized. The command verifies that the resulting features have the
512 dimensions required by AAG.

### 4. Train AAG

```bash
python AAG_training.py \
  --dataset-path "./data/emotions_aag.pt" \
  --output-dir "./checkpoint" \
  --device auto
```

## CLIP-AAG method

### Stage 1: fine-tune CLIP

CLIP is fine-tuned using only images from seen emotion classes. For an image
and its emotion prompt, the visual and text encoders produce paired features.
Training uses the symmetric image-to-text and text-to-image contrastive
cross-entropy objective.

The default prompt reproduces the released implementation:

```text
A picture seems to express some feelings like {label}.
```

### Stage 2: train AAG

The fine-tuned CLIP model is frozen and used to produce 512-dimensional image
and text embeddings. Images belonging to each seen emotion are averaged into a
class-level visual prototype. AAG then learns to transform the corresponding
text embedding toward the visual prototype.

The released AAG settings match the paper:

- generator: `512 -> 256 -> 128 -> 256 -> 512`
- LeakyReLU activations with intermediate and output skip connections
- critic: `512 -> 256 -> 1` with ReLU
- RMSprop learning rate: `1e-5`
- epochs: `500`
- critic-generator update ratio: `1:1`
- critic weight clipping: `[-0.01, 0.01]`

At inference, candidate unseen-class text embeddings are transformed by AAG.
The predicted emotion is the candidate whose generated attribute has the
largest dot product with the unseen image embedding.

## AAG dataset format

Provide the full path to one preprocessed PyTorch `.pt` file. The file must
contain a dictionary with these keys:

```python
{
    "seen_img": [...],
    "seen_text": [...],
    "seen_label": [...],
    "unseen_img": [...],
    "unseen_text": [...],
    "unseen_label": [...],
}
```

Each image and text embedding must be a one-dimensional vector of length 512.
Within each partition, the image, text, and label collections must have equal
lengths.

The trainer does not accept a split number. If you use multiple dataset splits,
preprocess each split as its own `.pt` file and pass the desired file's full
path.

> **Security:** PyTorch dataset files may use Python pickle internally. Only
> load `.pt` files from sources you trust.

## Training from the command line

The only required argument is the dataset path:

```bash
python AAG_training.py \
  --dataset-path "/absolute/path/to/your/dataset.pt"
```

Example with optional settings:

```bash
python AAG_training.py \
  --dataset-path "/absolute/path/to/your/dataset.pt" \
  --output-dir "./checkpoint" \
  --epochs 500 \
  --batch-size 4 \
  --lr 1e-5 \
  --n-critic 1 \
  --seed 42 \
  --device auto
```

Available options:

- `--dataset-path` — required full path to the preprocessed `.pt` file
- `--output-dir` — output directory; defaults to `./checkpoint`
- `--epochs` — training epochs; defaults to `500`
- `--batch-size` — class-prototype batch size; defaults to `4`
- `--lr` — RMSprop learning rate; defaults to `1e-5`
- `--n-critic` — critic steps between generator updates; defaults to `1`
- `--seed` — PyTorch and data-shuffling seed; defaults to `42`
- `--device` — `auto`, `cpu`, or `cuda`; defaults to `auto`

Run `python AAG_training.py --help` for the complete CLI reference.

## Fine-tuning CLIP

`finetune_clip.py` accepts the full path to one `.npz` manifest; there is no
dataset split number or hard-coded machine path.

The `.npz` file must contain a one-dimensional `seen` array of image paths:

```python
import numpy as np

np.savez(
    "fine_tuning_data.npz",
    seen=np.array(
        [
            "/data/happy/image_001.jpg",
            "/data/sad/image_002.jpg",
        ]
    ),
)
```

The parent directory of each image becomes its label. With the default prompt
template, `/data/happy/image_001.jpg` becomes:

```text
A picture seems to express some feelings like happy.
```

Run fine-tuning with:

```bash
python finetune_clip.py \
  --dataset-path "/absolute/path/to/fine_tuning_data.npz" \
  --output-path "./models/clip_finetuned_emotions.pt" \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-6 \
  --weight-decay 1e-6 \
  --device auto
```

Useful options include:

- `--model-name` — OpenAI CLIP architecture; defaults to `ViT-B/16`
- `--prompt-template` — custom prompt containing `{label}`
- `--num-workers` — image-loading worker count; defaults to `0`
- `--seed` — training and shuffle seed; defaults to `7`
- `--download-root` — optional directory for downloaded CLIP weights

If `--output-path` is omitted, the checkpoint is saved as
`./models/clip_finetuned_<dataset-name>.pt`. It contains the model state,
optimizer state, model name, prompt template, final loss, and per-epoch losses.

The `unseen` array from the research notebook is not needed for fine-tuning and
is therefore optional. Evaluation remains separate from training.

## Predicting with a saved model

`AAG_evaluate.py` accepts a saved checkpoint, one or more test image
embeddings, and labeled candidate text embeddings:

```bash
python AAG_evaluate.py \
  --checkpoint-path "./checkpoint/wgan_prototype_my_dataset.pth" \
  --test-image "/absolute/path/to/test_image.npy" \
  --test-text "/absolute/path/to/candidate_text.pt" \
  --top-k 3
```

The image input can be:

- a single `(512,)` embedding
- a batch shaped `(N, 512)`
- a `.npy`, `.npz`, `.pt`, `.pth`, `.json`, `.csv`, or `.txt` file

The candidate text file can be a `.pt`, `.pth`, `.json`, or `.npz` file in
either of these forms:

```python
# Explicit labels and embeddings
{
    "labels": ["cat", "dog"],
    "embeddings": torch.stack([cat_text_embedding, dog_text_embedding]),
}

# Direct label-to-embedding mapping
{
    "cat": cat_text_embedding,
    "dog": dog_text_embedding,
}
```

For example, save a test image embedding and candidate text embeddings with:

```python
import numpy as np
import torch

np.save("test_image.npy", image_embedding)
torch.save(
    {
        "labels": ["cat", "dog"],
        "embeddings": torch.stack([cat_text_embedding, dog_text_embedding]),
    },
    "candidate_text.pt",
)
```

Files containing `unseen_label` and `unseen_text` are also accepted directly.
If one `.pt` file contains `unseen_img`, `unseen_text`, and `unseen_label`, the
same file can be passed to both input arguments:

```bash
python AAG_evaluate.py \
  --checkpoint-path "./checkpoint/wgan_prototype_my_dataset.pth" \
  --test-image "/absolute/path/to/test_dataset.pt" \
  --test-text "/absolute/path/to/test_dataset.pt"
```

Use `--json-output results.json` to save the ranked predictions.

Both image and text inputs must already be 512-dimensional embeddings from the
same encoder used to prepare the training data. A raw JPG and plain text cannot
be passed directly to AAG without first applying that encoder.

## What the trainer does

1. Validates the dataset keys, collection lengths, labels, and embedding shapes.
2. Groups seen image embeddings by class.
3. Averages each class's image embeddings into one visual prototype.
4. Pairs each visual prototype with its class text embedding.
5. Trains the generator and discriminator with the original AAG objective.
6. Evaluates top-1 zero-shot accuracy on the unseen examples after each epoch.
7. Saves the generator checkpoint with the best unseen accuracy.

## Outputs

For a dataset named `my_dataset.pt`, the output directory contains:

- `wgan_prototype_my_dataset.pth` — best generator state dictionary
- `LOG-my_dataset.txt` — best zero-shot accuracy and its epoch

The dataset filename, rather than a manually supplied split number, is used in
the output names.

## Reproducibility notes

- Always pass the exact unseen emotion classes specified by the selected paper
  split; the preprocessor intentionally provides no random split option.
- CLIP fine-tuning and AAG gradient updates use seen classes only.
- AAG monitors unseen zero-shot accuracy after each epoch and retains the
  checkpoint with the best unseen performance, following the paper.
- Keep the prompt template and CLIP model consistent between fine-tuning,
  embedding extraction, and inference.
- Run each FEAr configuration in a separate output directory to avoid
  overwriting checkpoints or logs.

## Citation

If this repository contributes to your work, please cite the manuscript:

```bibtex
@misc{hew2026beyond,
  title  = {Beyond Illusions of Competence: Revisiting Zero-Shot Learning
            in Emotion Recognition with FEAr Dataset},
  author = {Hew, Zhong Ken and Wong, Lai-Kuan and Chan, Chee Seng},
  year   = {2026},
  note   = {Accepted at Signal Processing: Image Communication}
}
```

Please also cite the datasets you use:

- Quanzeng You, Jiebo Luo, Hailin Jin, and Jianchao Yang. *Building a Large
  Scale Dataset for Image Emotion Recognition: The Fine Print and the
  Benchmark.* AAAI, 2016. (FI)
- Kuan-Chuan Peng, Tsuhan Chen, Amir Sadovnik, and Andrew C. Gallagher. *A
  Mixed Bag of Emotions: Model, Predict, and Transfer Emotion Distributions.*
  CVPR, 2015.
  ([Emotion6 download page](http://chenlab.ece.cornell.edu/downloads.html))
- Jana Machajdik and Allan Hanbury. *Affective Image Classification Using
  Features Inspired by Psychology and Art Theory.* ACM Multimedia, 2010.
  ([ArtPhoto download page](https://www.imageemotion.org/))
