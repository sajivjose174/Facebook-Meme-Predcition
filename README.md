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

Meme Text
    ↓
BERT
    ↓
[CLS] Representation (768)
    ↓
Classification Layer
    ↓
Hateful / Not Hateful

### 2. ResNet50 — Image Classification

Meme Image
    ↓
ResNet50
    ↓
2048-D Image Features
    ↓
Dense Classifier
    ↓
Hateful / Not Hateful


### 3.BERT + ResNet50 — Multimodal Fusion

Text → BERT → 768-D
                  \
                   → Concatenate → 2816-D → Classifier
                  /
Image → ResNet → 2048-D


### 4. CLIP — Multimodal Representation

Text → CLIP Text Encoder → 512-D
                              \
                               → 1024-D → Classifier
                              /
Image → CLIP Vision Encoder → 512-D



| Model | Accuracy | Precision | Recall | F1 Score | AUROC |
|---|---:|---:|---:|---:|---:|
| BERT | 0.5500 | 0.6263 | 0.2480 | 0.3553 | 0.6359 |
| ResNet50 | 0.5360 | — | — | — | — |
| BERT + ResNet50 | **0.5740** | **0.6697** | **0.2920** | **0.4067** | 0.6243 |
| CLIP | 0.5500 | 0.6374 | 0.2320 | 0.3402 | **0.6395** |


The results show that the models were able to learn some discriminative signal, but performance on the hateful class remained limited.
In particular, the CLIP model achieved an AUROC of 0.6395, while its recall for the hateful class was 23.2%.
These results motivated further error analysis to understand where the models struggled.


##Error Analysis:
Error analysis was performed on the validation predictions, with particular attention to false negatives and false positives.
The CLIP model produced:
- 192 False Negatives
- 33 False Positives
Several false-negative examples contained hateful or mocking content but were classified as non-hateful with high confidence.
This highlights the difficulty of detecting hateful meaning when interpretation depends on the interaction between text, visual context and the underlying message.
Some representative examples are included in the project notebooks.
Note: Some examples from the original dataset contain offensive or sensitive content. They are included only for model evaluation and error analysis.


##Technologies Used:
Python
TensorFlow / Keras
KerasHub
Hugging Face Transformers
BERT
ResNet50
CLIP
NumPy
Pandas
Scikit-learn
Matplotlib
Google Colab

##Project Structure:
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

##Key Learnings:
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
One of the key findings was that combining modalities does not automatically guarantee better performance. The effectiveness of multimodal learning depends on how information from the different modalities is represented, aligned and fused.


##Future Improvements
Potential improvements include:
- Fine-tuning CLIP instead of using frozen representations
- Joint fine-tuning of BERT and ResNet
- Attention-based multimodal fusion
- OCR for text embedded within images
- More advanced vision-language models
- Hyperparameter optimization
- More extensive error analysis
- Improved handling of class imbalance and false negatives.


##References
- Facebook/Meta Hateful Memes Dataset
- BERT: Bidirectional Encoder Representations from Transformers
- ResNet: Deep Residual Learning for Image Recognition
- CLIP: Contrastive Language–Image Pre-training
- TensorFlow / Keras
- KerasHub
- Hugging Face Transformers



