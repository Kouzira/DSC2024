# Multimodal Sarcasm Detection on Vietnamese Social Media

End-to-end multimodal classifier for the **UIT/VLSP 2024 ViMMSD** task — detecting
sarcasm in Vietnamese social-media posts that pair a Vietnamese caption with an
image (often containing embedded text).

Built around the `Building a Bridge` approach (Wang et al., NLPBT 2020) and extended
with **cascaded two-stage text fusion** to handle the dual text input (image OCR +
post caption) that is specific to ViMMSD.

> **Status:** code is bug-fixed and CV-ready; results table below is filled in
> after a clean baseline run on the train split (see [Reproduce](#reproduce)).

---

## Task

Classify each `(image, caption)` post into one of 4 labels:

| Label            | Definition                                                   |
|------------------|--------------------------------------------------------------|
| `multi-sarcasm`  | Sarcasm arises from the image–caption combination            |
| `not-sarcasm`    | Neither image nor caption is sarcastic                       |
| `image-sarcasm`  | Sarcasm comes from the image alone                           |
| `text-sarcasm`   | Sarcasm comes from the caption alone                         |

Dataset is heavily imbalanced (`multi-sarcasm` dominates) and contains
~10 K training samples.

---

## Architecture

```
                                                 ┌──────────────────────────────┐
       Image                                     │  Bridge layer  (1x1 conv ≡   │
        │                                        │  nn.Linear over channels)    │
        ▼                                        │  2048 → 768 + GELU           │
   ┌─────────────────┐                           └───────────────┬──────────────┘
   │ EfficientNet-B5 │──► last_hidden_state                      │
   │   (pretrained)  │    (B, 2048, 15, 15)                      │
   └─────────────────┘    + pooler_output                        ▼
        │              (B, 2048, 1, 1)             ┌─────────────────────────┐
        │       AvgPool2d(2) + flatten             │ visual prefix tokens    │
        └────────────────────────────────────────► │ (B, 65, 768)            │
                                                   └──────────┬──────────────┘
                                                              │
   OCR text  ──► tokenize  ──► input_ids (B, 191)             │
                                                              ▼
                                       ┌─────────────────────────────────────┐
                                       │  PhoBERT-base-v2 (custom_forward)   │
                                       │  Prepends visual prefix to text     │
                                       │  embeddings; attention_mask derived │
                                       │  from pad tokens                    │
                                       └────────────────┬────────────────────┘
                                                        │
                              AvgPool1d(4) + bridge_2   │
                              (sequence-level pooling)  ▼
                                                ┌────────────────┐
                                                │ (B, 64, 768)   │ ◄── "visual"
                                                │ feature prefix │     prefix for
                                                │ for stage 2    │     PhoBERT-2
                                                └───────┬────────┘
                                                        │
   Caption  ──► tokenize  ──► input_ids (B, 224)        │
                                                        ▼
                                       ┌─────────────────────────────────────┐
                                       │  PhoBERT-base-v2 (custom_forward)   │
                                       │  Same prefix-injection scheme       │
                                       └────────────────┬────────────────────┘
                                                        │ pooler outputs
                                                        ▼
                                        concat(stage1.pool, stage2.pool)
                                                  (B, 1536)
                                                        │
                                                        ▼
                                        ┌─────────────────────────────────┐
                                        │  fc1 → GELU → dropout(0.5) → fc2│
                                        │       (B, 1536) → (B, 4) logits │
                                        └─────────────────────────────────┘
```

**Key design choices**

- **Prefix-style visual injection.** Visual features are projected to PhoBERT's
  hidden dim and prepended to text embeddings — letting PhoBERT's self-attention
  do cross-modal mixing for free, without expensive image-text pretraining.
- **Cascaded two-stage fusion.** ViMMSD has two text sources (OCR text scraped
  from the image, and the post caption) with different distributions. Stage 1
  fuses image + OCR; stage 2 takes a downsampled stage-1 sequence as its own
  "visual" prefix together with the caption. Both `pooler_output`s are then
  concatenated for the classifier.
- **Pad-aware attention.** Padding tokens of OCR/caption are correctly excluded
  from self-attention by deriving an `attention_mask` from `pad_token_id` and
  concatenating it with an all-ones mask for the visual prefix.

---

## Mapping: paper → this code

| Wang et al. 2020 ("Building a Bridge")        | This implementation                                     |
|-----------------------------------------------|---------------------------------------------------------|
| Text encoder: BERT (English)                  | **PhoBERT-base-v2** (Vietnamese)                        |
| Image encoder: ResNet, output `(7, 7) + (1,1)`| **EfficientNet-B5**, output `(15, 15) + (1, 1)`         |
| Bridge layer: `1×1 conv` for channel projection| `nn.Linear` over channel dim (mathematically identical) |
| Flatten spatial map → token sequence          | `torch.flatten(...).permute(...)` after `AvgPool2d(2)`  |
| Single text input                             | **Two text inputs (OCR + caption) in cascaded stages**  |
| Multi-head attention for cross-modal mixing   | Self-attention in PhoBERT after prefix concat           |

**Original contributions in this repo** (beyond the paper):
1. Vietnamese adaptation with PhoBERT + EasyOCR + VnCoreNLP segmentation.
2. **Cascaded two-stage transformer fusion** for joint OCR + caption modeling.
3. Reproducible training pipeline: seeded splits, F1-driven checkpoint selection,
   per-epoch metric logging.

---

## Results

Baseline run on the official ViMMSD train split (85/15 train/val), seed 42:

| Metric         | Value      |
|----------------|------------|
| Macro F1       | TBD        |
| Accuracy       | TBD        |
| F1 `multi-sarcasm` | TBD    |
| F1 `not-sarcasm`   | TBD    |
| F1 `image-sarcasm` | TBD    |
| F1 `text-sarcasm`  | TBD    |

### Ablation studies (planned)

| Variant                                                | Macro F1 |
|--------------------------------------------------------|----------|
| Full model (image + OCR + caption, cascaded)           | TBD      |
| – without image (text-only baseline)                   | TBD      |
| – without OCR (image + caption only)                   | TBD      |
| – without caption (image + OCR only)                   | TBD      |
| Single PhoBERT with concat(OCR, caption) instead of cascade | TBD |
| Class-weighted CrossEntropy vs vanilla CE              | TBD      |

---

## Reproduce

### 1. Install

Requires Python ≥ 3.10, a CUDA-capable GPU (≥ 12 GB), and Java (for VnCoreNLP).

```bash
pip install -r requirements.txt
# Linux only:
sudo apt install openjdk-21-jdk openjdk-21-jre -y
# Windows: install a JDK and ensure `java` is on PATH.
```

### 2. Datasets

Two Kaggle datasets are downloaded automatically by `utils.download_dataset()`:

- `tmaitn/uitdsc24-train-dataset` — preprocessed train set (image + OCR.pt + caption.pt + label)
- `longnguynvhong/dsc2024-public-test` — public test set for inference

Authenticate with `kagglehub` (`~/.kaggle/kaggle.json` or `kagglehub.login()`).

### 3. Train

```bash
cd src
python train.py
```

Outputs (under `~/DSC2024/checkpoint/`):

- `checkpoint_0.pth` — rolling latest checkpoint
- `best_model.pth` — best by macro F1
- `history.json` — train / val loss curves
- `metrics.json` — full per-epoch metric history (use for plotting)

### 4. Inference on public test

Runs at the end of `train.py` automatically, writing
`~/DSC2024/results.json` in the official submission format.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── preprocess_train_data.ipynb   # one-off data prep pipeline
├── vimmsd-warmup.json            # sample annotation
└── src/
    ├── data.py        # MultiMediaDataset
    ├── model.py       # ImageFeatureExtractor, ModifiedPhoBERT, MultiModalClassifier
    ├── train.py       # training loop, evaluation, inference
    └── utils.py       # metrics, optimizer, lr scheduler, checkpoint I/O,
                        # lazy-init OCR / tokenizer / segmenter, EarlyStopping
```

---

## References

- **Wang, X., Sun, X., Yang, T., & Wang, H. (2020).** Building a Bridge: A
  Method for Image-Text Sarcasm Detection Without Pretraining on Image-Text
  Data. *Proceedings of the First International Workshop on NLP Beyond Text*
  (NLPBT 2020), pp. 19–29. [aclanthology.org/2020.nlpbt-1.3](https://aclanthology.org/2020.nlpbt-1.3)
- **Nguyen, D. Q., & Tuan Nguyen, A. (2020).** PhoBERT: Pre-trained language
  models for Vietnamese. *Findings of EMNLP 2020*.
- **Tan, M., & Le, Q. V. (2019).** EfficientNet: Rethinking Model Scaling for
  Convolutional Neural Networks. *ICML 2019*.

---

## Engineering notes

This repo started as a competition submission and was refactored into a
portfolio-quality codebase. Notable cleanups (see git history for full
context):

- Removed an unused 1700-file fork of HuggingFace `transformers` (~45 MB).
- Fixed 5 silent bugs that were degrading training quality:
  caption-overwrite-by-OCR at inference, padding tokens attended to as real
  tokens, softmax applied before `CrossEntropyLoss`, `notfoundcnt` undefined,
  `load_checkpoint` discarding the saved epoch.
- Replaced top-level GPU side-effects with lazy initializers (OCR / tokenizer
  / segmenter), so DataLoader workers don't re-allocate them.
- Cross-platform paths (no more `/tmp` or `/root` hardcodes).
- Added macro F1 / per-class F1 / confusion matrix tracking and best-by-F1
  checkpoint selection.
