# Facebook Meme Prediction

A multimodal deep learning project for detecting hateful memes using both textual and visual information from the Facebook Hateful Memes dataset.

## 📌 Project Overview

Memes often communicate meaning through a combination of text and images. A meme that appears harmless from its text alone may have a completely different meaning when combined with its visual context.

This project explores whether combining Natural Language Processing (NLP) and Computer Vision representations can improve hateful meme classification.

Multiple approaches were implemented and compared:

- **BERT** — Text-only classification
- **ResNet50** — Image-only classification
- **BERT + ResNet50** — Multimodal feature fusion
- **CLIP** — Multimodal image-text representation

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

## 🧠 Model Architectures

The project evaluates four different approaches for hateful meme classification.

### Overall Architecture

    FACEBOOK HATEFUL MEMES
              |
      +-------+-------+
      |       |       |
      v       v       v
     TEXT   IMAGE  TEXT + IMAGE
      |       |       |
      v       v       v
    BERT  ResNet50  BERT + ResNet50
      |       |       |
    768-D  2048-D   2816-D
      |       |       |
      v       v       v
 Classifier Classifier Fusion Classifier
      |       |       |
      +-------+-------+
              |
              v
      Hateful / Not Hateful


### CLIP Architecture

    TEXT                         IMAGE
      |                            |
      v                            v
    CLIP Text                 CLIP Vision
    Encoder                     Encoder
      |                            |
      v                            v
    512-D                        512-D
      \                            /
       \                          /
        +------ Concatenate -----+
                    |
                    v
                 1024-D
                    |
                    v
              Dense Layer (256)
                    |
                    v
                  ReLU
                    |
                    v
              Dropout (0.3)
                    |
                    v
                Output
                    |
                    v
          Hateful / Not Hateful

### 1. BERT — Text-Only Classification

BERT was used to process the textual content of each meme.

The `[CLS]` representation was extracted from BERT and used as the input to a binary classification layer.

BERT produces a **768-dimensional contextual representation** for the text.

### 2. ResNet50 — Image-Only Classification

A pretrained ResNet50 model was used to extract visual features from the meme images.

The original ImageNet classification head was removed and the network was used as a visual feature extractor.

This produced **2048-dimensional image features**, which were passed to a small classification network.

### 3. BERT + ResNet50 — Multimodal Classification

The BERT text representation and ResNet50 image representation were combined using feature-level fusion.

- BERT representation: **768 dimensions**
- ResNet50 representation: **2048 dimensions**
- Combined representation: **2816 dimensions**

The combined representation was passed through:

`Dense(256) → ReLU → Dropout(0.3) → Output`

### 4. CLIP — Multimodal Classification

CLIP was used to obtain jointly learned representations for the text and image modalities.

The CLIP text and vision encoders produced:

- Text embedding: **512 dimensions**
- Image embedding: **512 dimensions**

These were concatenated into a **1024-dimensional multimodal representation** and passed through a small classification network.

The CLIP backbone was frozen during feature extraction, while the downstream classifier was trained for the hateful meme classification task.

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | AUROC |
|---|---:|---:|---:|---:|---:|
| BERT | 0.5500 | 0.6263 | 0.2480 | 0.3553 | 0.6359 |
| ResNet50 | 0.5360 | — | — | — | — |
| BERT + ResNet50 | **0.5740** | **0.6697** | **0.2920** | **0.4067** | 0.6243 |
| CLIP | 0.5500 | 0.6374 | 0.2320 | 0.3402 | **0.6395** |

### CLIP Confusion Matrix

    [[217  33]
     [192  58]]

The CLIP model produced:

- True Negatives: **217**
- False Positives: **33**
- False Negatives: **192**
- True Positives: **58**

### Key Observations

The models learned meaningful discriminative signal, but performance on the hateful class remained limited.

The CLIP model achieved an **AUROC of 0.6395**, with a hateful-class recall of **23.2%**.

The BERT + ResNet50 model achieved the highest F1 score among the evaluated models at **0.4067**.

These results demonstrate that simply combining pretrained representations does not automatically guarantee improved performance. The effectiveness of multimodal learning depends on how information from different modalities is represented, aligned and fused.

---

## 🔍 Error Analysis

Error analysis was performed on the validation predictions, with particular attention to false negatives and false positives.

The CLIP model produced:

- **192 False Negatives**
- **33 False Positives**

Several false-negative examples contained hateful or mocking content but were classified as non-hateful with high confidence.

This highlights the difficulty of detecting hateful meaning when interpretation depends on the interaction between text, visual context and the underlying message.

Some representative examples are included in the project notebooks.

> **Note:** Some examples from the original dataset contain offensive or sensitive content. They are included only for model evaluation and error analysis.

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- KerasHub
- Hugging Face Transformers
- BERT
- ResNet50
- CLIP
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Google Colab

---

## 📂 Project Structure

    Facebook-Meme-Prediction/
    │
    ├── notebooks/
    │   ├── EDA
    │   ├── BERT
    │   ├── ResNet50
    │   ├── BERT_ResNet_Multimodal
    │   └── CLIP
    │
    ├── README.md
    └── requirements.txt

---

## 🎯 Key Learnings

This project provided practical experience with:

- Text classification using BERT
- Transfer learning with ResNet50
- Multimodal feature fusion
- Vision-language models
- CLIP embeddings
- Binary classification
- AUROC and classification metrics
- Confusion matrix analysis
- Error analysis
- Working with pretrained deep learning models

One of the key findings was that **combining modalities does not automatically guarantee better performance**. The effectiveness of multimodal learning depends on how information from the different modalities is represented, aligned and fused.

---

## 🚀 Future Improvements

Potential improvements include:

- Fine-tuning CLIP instead of using frozen representations
- Joint fine-tuning of BERT and ResNet
- Attention-based multimodal fusion
- OCR for text embedded within images
- More advanced vision-language models
- Hyperparameter optimization
- More extensive error analysis
- Improved handling of false negatives
- Exploring stronger multimodal fusion strategies

---

## 📚 References

- Facebook/Meta Hateful Memes Dataset
- BERT: Bidirectional Encoder Representations from Transformers
- ResNet: Deep Residual Learning for Image Recognition
- CLIP: Contrastive Language–Image Pre-training
- TensorFlow / Keras
- KerasHub
- Hugging Face Transformers
