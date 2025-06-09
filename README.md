# Duplicate Question Detector

A complete NLP-based pipeline to detect whether two questions are duplicates. Inspired by the Quora Question Pair challenge, this project began with basic ML logic and gradually evolved into a robust BiLSTM deep learning model with Streamlit frontend and clean data preprocessing.

---

## Table of Contents

- [Overview](#overview)
- [How the Project Evolved](#how-the-project-evolved)
- [Modeling Approaches](#modeling-approaches)
- [Dataset](#dataset)
- [Setup Instructions](#setup-instructions)
- [Streamlit Usage](#streamlit-usage)
- [Resolved Issues](#resolved-issues)
- [Future Improvements](#future-improvements)

---

## Overview

The goal is to predict if two given questions are semantically similar. This helps platforms like Quora avoid duplicate content and improve user search experience.

---

## How the Project Evolved

| Phase | Approach | Key Milestone |
|-------|----------|---------------|
| 🔹 Phase 1 | **Rule-Based Baseline** | Handcrafted logic using word overlap, Jaccard similarity |
| 🔹 Phase 2 | **TF-IDF + ML Models** | Implemented Logistic Regression, XGBoost, improved accuracy |
| 🔹 Phase 3 | **Word2Vec + ML** | Used pre-trained GloVe vectors for better semantic capture |
| 🔹 Phase 4 | **BiLSTM Deep Learning** | Built using Keras for superior performance on text sequences |
| 🔹 Phase 5 | **Streamlit App** | UI for real-time predictions |
| 🔹 Phase 6 | **Deployment Attempts** | Faced and overcame Python/TensorFlow version issues |

---

## 🔬 Modeling Approaches

### 1. Baseline (Manual Logic)
- Jaccard similarity
- Common word count
- Heuristic rules

### 2. TF-IDF + ML
- `TfidfVectorizer` on both questions
- Concatenation, difference, cosine similarity
- Trained: Logistic Regression, RandomForest, XGBoost

### 3. Word2Vec + ML
- Loaded GloVe embeddings
- Averaged word vectors
- Distance-based features (cosine, Manhattan, Euclidean)

### 4. Deep Learning (BiLSTM)
- Tokenized & padded sequences
- Model:
  - Embedding → BiLSTM → Dense
- Trained with Binary Crossentropy & Adam optimizer

---

## Dataset

- **Source**: [Quora Question Pair](https://www.kaggle.com/c/quora-question-pairs/data)
- **Size**: 404K+ rows
- **Columns**: `question1`, `question2`, `is_duplicate`

---

## 🛠 Setup Instructions

### Clone & Install

```bash
git clone https://github.com/kunalpunia94/duplicate-question-detector
cd duplicate-question-detector
pip install -r requirements.txt
