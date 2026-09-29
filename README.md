# Facebook-Meme-Predcition
Multimodal hateful meme classification using BERT, ResNet50, and CLIP, with model comparison and error analysis on the Facebook Hateful Memes dataset.

## 📌 Project Overview

Memes often communicate meaning through a combination of text and images. A meme that appears harmless from its text alone may have a completely different meaning when combined with its visual context.

This project explores whether combining Natural Language Processing (NLP) and Computer Vision representations can improve hateful meme classification.

Multiple approaches were implemented and compared:

- BERT — Text-only classification
- ResNet50 — Image-only classification
- BERT + ResNet50 — Multimodal feature fusion
- CLIP — Multimodal image-text representation

The primary evaluation metric for the dataset is **AUROC**, along with Accuracy, Precision, Recall and F1-score.

---

## 📊 Dataset

The project uses the **Facebook/Meta Hateful Memes Dataset**.

Each sample contains:

- Meme image
- Meme text
- Binary label

Labels:

- `0` — Not Hateful
- `1` — Hateful

### Dataset Distribution

The training dataset contains **8,500 samples**:

| Class | Samples | Percentage |
|---|---:|---:|
| Not Hateful | 5,450 | 64.1% |
| Hateful | 3,050 | 35.9% |

The validation set used in this project contains **500 samples**, with 250 examples from each class.

---

## 🧠 Models

### 1. BERT — Text Classification

BERT was used to extract contextual representations from the meme text.

```text
Meme Text
    ↓
BERT
    ↓
[CLS] Representation (768)
    ↓
Classification Layer
    ↓
Hateful / Not Hateful
