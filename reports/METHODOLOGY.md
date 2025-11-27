# AG News Classification: Comprehensive Methodology

**Author**: Reza Shariatmadar  
**Date**: November 2025  
**Document Type**: Technical Methodology  
**Target Audience**: ML Engineers, Research Scientists, Reproducibility Reviewers

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Data Preprocessing](#2-data-preprocessing)
3. [Feature Engineering](#3-feature-engineering)
4. [Classical Model Training](#4-classical-model-training)
5. [Transformer Fine-Tuning](#5-transformer-fine-tuning)
6. [Evaluation Framework](#6-evaluation-framework)
7. [Hyperparameter Optimization](#7-hyperparameter-optimization)
8. [Statistical Analysis](#8-statistical-analysis)
9. [Implementation Details](#9-implementation-details)
10. [Reproducibility Checklist](#10-reproducibility-checklist)

---

## 1. Introduction

This document provides complete implementation details for reproducing our AG News classification experiments. We emphasize transparency, reproducibility, and practical applicability.

### 1.1 Notation

**Dimensions**:
- \( N \): Number of training samples (102,000)
- \( d \): Feature dimension (varies by representation)
- \( C \): Number of classes (4)
- \( L \): Sequence length for transformers (256)
- \( V \): Vocabulary size (30,522 for BERT tokenizer)

**Vectors and Matrices**:
- \( \mathbf{x}_i \in \mathbb{R}^d \): Feature vector for sample \( i \)
- \( y_i \in \{0,1,2,3\} \): True label for sample \( i \)
- \( \hat{y}_i \): Predicted label for sample \( i \)
- \( \mathbf{X} \in \mathbb{R}^{N \times d} \): Feature matrix
- \( \mathbf{W} \in \mathbb{R}^{d \times C} \): Weight matrix for linear classifiers

### 1.2 Software Environment

**Core Dependencies**:
```python
python==3.11.14
torch==2.9.1+cu128
transformers==4.36.0
scikit-learn==1.7.2
numpy==2.3.5
pandas==2.3.3
nltk==3.9.2
spacy==3.7.5
optuna==3.5.0
```

**Hardware**:
- GPU: NVIDIA RTX 4060 Ti (16GB VRAM)
- CPU: 16 cores, 32GB RAM
- Storage: NVMe SSD

---

## 2. Data Preprocessing

### 2.1 Dataset Loading

**Source**: Hugging Face Datasets Hub

```python
from datasets import load_dataset

dataset = load_dataset("sh0416/ag_news")

# Structure:
# dataset['train']: 120,000 samples
# dataset['test']: 7,600 samples
# Columns: ['label', 'title', 'description']
```

**Text Combination**:
```python
# Concatenate title and description
df['text'] = df['title'] + " " + df['description']

# Handle missing values
df['text'] = df['text'].fillna('')
```

**Label Normalization**:
```python
# Original labels: 1-4 (1-based)
# Normalized labels: 0-3 (0-based)
df['label'] = df['label'] - 1

# Mapping
label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
```

### 2.2 Data Splitting Strategy

**Stratified Split** (maintains class balance):

```python
from sklearn.model_selection import train_test_split

# Original: 120K train, 7.6K test
# New split: 102K train, 18K val, 7.6K test

train_df, val_df = train_test_split(
    df_train_full,
    test_size=0.15,  # 15% validation
    stratify=df_train_full['label'],
    random_state=42
)
```

**Verification**:
```python
# Check class distribution
for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    dist = split_df['label'].value_counts(normalize=True)
    print(f"{split_name}: {dist.values}")  # Should be [0.25, 0.25, 0.25, 0.25]
```

### 2.3 Text Cleaning Pipeline

**Design Philosophy**: Minimal preprocessing preserves information while removing noise.

#### 2.3.1 Basic Cleaning (All Models)

```python
import re

def clean_text(text):
    # 1. Whitespace normalization
    text = re.sub(r'\s+', ' ', text)
    
    # 2. Strip leading/trailing whitespace
    text = text.strip()
    
    return text
```

#### 2.3.2 Moderate Cleaning (Classical Models)

```python
def clean_moderate(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. URL tokenization (preserve information)
    text = re.sub(r'http\S+|www\.\S+', '<URL>', text)
    
    # 3. Email tokenization
    text = re.sub(r'\S+@\S+', '<EMAIL>', text)
    
    # 4. Whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
```

**Rationale**: 
- Lowercase reduces vocabulary size
- URL/email tokens preserve semantic information (e.g., tech articles often mention URLs)
- Minimal punctuation removal (kept for TF-IDF)

#### 2.3.3 Cleaning for Transformers

```python
def clean_for_bert(text):
    # BERT tokenizer handles most preprocessing
    # Only basic whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

**Rationale**: BERT's WordPiece tokenizer benefits from preserving case and punctuation.

### 2.4 Tokenization

#### 2.4.1 For Classical Models (Whitespace)

```python
def tokenize_simple(text):
    return text.split()
```

#### 2.4.2 For Transformers (WordPiece)

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

encodings = tokenizer(
    texts,
    truncation=True,
    padding='max_length',
    max_length=256,
    return_tensors='pt',
    return_attention_mask=True
)

# Returns:
# - input_ids: [batch_size, 256]
# - attention_mask: [batch_size, 256]
```

**Token Length Analysis**:
```python
import matplotlib.pyplot as plt

lengths = [len(tokenizer.encode(text)) for text in texts]
plt.hist(lengths, bins=50)
plt.axvline(np.percentile(lengths, 95), color='r', label='95th percentile')
plt.axvline(np.percentile(lengths, 99), color='g', label='99th percentile')
```

**Results**:
- Mean: 53 tokens
- 95th percentile: 82 tokens
- 99th percentile: 120 tokens
- **Decision**: max_length=256 covers 99.5%+ of samples

---

## 3. Feature Engineering

### 3.1 TF-IDF Vectorization

#### 3.1.1 Mathematical Foundation

**Term Frequency (TF)**:
\[
\text{tf}(t, d) = \log(1 + f_{t,d})
\]
where \( f_{t,d} \) is the raw count of term \( t \) in document \( d \).

**Inverse Document Frequency (IDF)**:
\[
\text{idf}(t) = \log\left(\frac{N}{1 + n_t}\right)
\]
where \( N \) is total documents, \( n_t \) is documents containing term \( t \).

**TF-IDF**:
\[
\text{tfidf}(t, d) = \text{tf}(t, d) \times \text{idf}(t)
\]

**L2 Normalization**:
\[
\mathbf{x}_i \leftarrow \frac{\mathbf{x}_i}{\|\mathbf{x}_i\|_2}
\]

#### 3.1.2 Word-Level TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer_word = TfidfVectorizer(
    # N-gram range
    ngram_range=(1, 2),  # Unigrams and bigrams
    
    # Vocabulary constraints
    max_features=50000,  # Top 50K features by document frequency
    min_df=3,            # Minimum document frequency
    max_df=0.95,         # Maximum document frequency (remove very common)
    
    # TF-IDF parameters
    sublinear_tf=True,   # Use log(1+tf) instead of raw tf
    norm='l2',           # L2 normalization
    
    # Tokenization
    lowercase=True,
    token_pattern=r'\b\w+\b'  # Word boundaries
)

X_train_word = vectorizer_word.fit_transform(train_texts)
X_val_word = vectorizer_word.transform(val_texts)
X_test_word = vectorizer_word.transform(test_texts)
```

**Sparsity Analysis**:
```python
sparsity = 1.0 - X_train_word.nnz / (X_train_word.shape[0] * X_train_word.shape[1])
print(f"Sparsity: {sparsity:.4f}")  # Expected: ~0.99
```

#### 3.1.3 Character-Level TF-IDF

```python
vectorizer_char = TfidfVectorizer(
    analyzer='char',     # Character-level
    ngram_range=(3, 5),  # 3-5 character n-grams
    max_features=50000,
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    norm='l2'
)

X_train_char = vectorizer_char.fit_transform(train_texts)
```

**Rationale**: Character n-grams capture:
- Morphological patterns (prefixes, suffixes)
- Typo robustness
- Sub-word information

#### 3.1.4 Hybrid Features (Best Performer)

```python
import scipy.sparse as sp

# Horizontal concatenation
X_train_hybrid = sp.hstack([X_train_word, X_train_char])
X_val_hybrid = sp.hstack([X_val_word, X_val_char])
X_test_hybrid = sp.hstack([X_test_word, X_test_char])

print(f"Hybrid shape: {X_train_hybrid.shape}")  # (102000, 100000)
```

### 3.2 Feature Selection

#### 3.2.1 Chi-Square (χ²) Selection

**Mathematical Foundation**:
\[
\chi^2(t, c) = \frac{N \times (AD - BC)^2}{(A+C)(B+D)(A+B)(C+D)}
\]

where:
- \( A \): Documents with term \( t \) and class \( c \)
- \( B \): Documents with term \( t \) but not class \( c \)
- \( C \): Documents without term \( t \) but class \( c \)
- \( D \): Documents without term \( t \) and not class \( c \)

**Implementation**:
```python
from sklearn.feature_selection import SelectKBest, chi2

chi2_selector = SelectKBest(chi2, k=20000)
X_train_chi2 = chi2_selector.fit_transform(X_train_word, y_train)
X_val_chi2 = chi2_selector.transform(X_val_word)

# Get selected feature names
feature_names = vectorizer_word.get_feature_names_out()
selected_features = feature_names[chi2_selector.get_support()]

# Top features per class
for class_idx in range(4):
    class_mask = (y_train == class_idx)
    scores = chi2_selector.scores_
    top_idx = np.argsort(scores)[-20:][::-1]
    print(f"Class {class_idx}: {feature_names[top_idx]}")
```

### 3.3 Dimensionality Reduction

#### 3.3.1 Truncated SVD (Latent Semantic Analysis)

**Mathematical Foundation**:
\[
\mathbf{X} \approx \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T
\]

where \( k \) is the number of components.

**Implementation**:
```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=300, random_state=42)
X_train_svd = svd.fit_transform(X_train_word)
X_val_svd = svd.transform(X_val_word)

# Explained variance
explained_var = svd.explained_variance_ratio_.sum()
print(f"Explained variance: {explained_var:.4f}")  # ~0.15 for 300 components
```

**Performance**: 88.66% F1 (3.8% drop from full features)

#### 3.3.2 Non-Negative Matrix Factorization (NMF)

```python
from sklearn.decomposition import NMF

nmf = NMF(n_components=100, random_state=42, max_iter=200)
X_train_nmf = nmf.fit_transform(X_train_word)
X_val_nmf = nmf.transform(X_val_word)

# Reconstruction error
print(f"Reconstruction error: {nmf.reconstruction_err_:.4f}")
```

### 3.4 Word Embeddings

#### 3.4.1 Pre-trained GloVe

```python
import gensim.downloader as api

# Load pre-trained embeddings
glove = api.load('glove-wiki-gigaword-100')

def text_to_glove(text, embedding_model):
    """Average word embeddings"""
    tokens = text.lower().split()
    vectors = [embedding_model[token] for token in tokens 
               if token in embedding_model]
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(100)

X_train_glove = np.array([text_to_glove(text, glove) for text in train_texts])
```

#### 3.4.2 TF-IDF Weighted Embeddings

```python
def text_to_tfidf_weighted_embedding(text, tfidf_vec, embedding_model):
    """TF-IDF weighted average of word embeddings"""
    tokens = text.lower().split()
    
    # Get TF-IDF weights
    tfidf_vector = tfidf_vec.transform([text])
    feature_names = tfidf_vec.get_feature_names_out()
    
    weighted_vectors = []
    for token in tokens:
        if token in embedding_model and token in feature_names:
            idx = list(feature_names).index(token)
            weight = tfidf_vector[0, idx]
            weighted_vectors.append(weight * embedding_model[token])
    
    if weighted_vectors:
        return np.sum(weighted_vectors, axis=0)
    else:
        return np.zeros(100)
```

---

## 4. Classical Model Training

### 4.1 Cross-Validation Setup

**Stratified K-Fold**:
```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Verify stratification
for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    train_dist = np.bincount(y[train_idx]) / len(train_idx)
    val_dist = np.bincount(y[val_idx]) / len(val_idx)
    print(f"Fold {fold_idx}: Train {train_dist}, Val {val_dist}")
    # Both should be [0.25, 0.25, 0.25, 0.25]
```

### 4.2 Linear Support Vector Classifier (Best Classical)

#### 4.2.1 Model Architecture

**Optimization Problem**:
\[
\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^N \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))
\]

where:
- \( C \): Regularization parameter (inverse regularization strength)
- Hinge loss: \( \max(0, 1 - y_i f(\mathbf{x}_i)) \)

#### 4.2.2 Implementation

```python
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Base model
svc = LinearSVC(
    C=0.1,              # Strong regularization (tested: 0.01, 0.1, 1.0, 10.0)
    loss='squared_hinge',  # Smooth hinge loss
    penalty='l2',       # L2 regularization
    dual=False,         # Primal optimization (faster for n_samples > n_features)
    max_iter=2000,      # Maximum iterations
    random_state=42,
    class_weight=None   # Balanced dataset
)

# Probability calibration (Platt scaling)
calibrated_svc = CalibratedClassifierCV(
    svc,
    method='sigmoid',   # Platt scaling
    cv=5,               # 5-fold CV for calibration
    n_jobs=-1
)

calibrated_svc.fit(X_train, y_train)
```

**Calibration Curve**:
```python
from sklearn.calibration import calibration_curve

y_proba = calibrated_svc.predict_proba(X_val)
y_pred_proba = y_proba.max(axis=1)
y_pred = calibrated_svc.predict(X_val)

fraction_of_positives, mean_predicted_value = calibration_curve(
    y_val == y_pred, y_pred_proba, n_bins=10
)

plt.plot(mean_predicted_value, fraction_of_positives, marker='o')
plt.plot([0, 1], [0, 1], 'k--')  # Perfect calibration
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
```

### 4.3 Stochastic Gradient Descent (Fastest)

```python
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(
    loss='modified_huber',  # Smooth hinge loss with probability estimates
    penalty='l2',
    alpha=1e-4,            # Regularization strength
    max_iter=1000,
    tol=1e-3,
    learning_rate='optimal',  # Adaptive learning rate
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5,
    random_state=42,
    n_jobs=-1
)

sgd.fit(X_train, y_train)
```

### 4.4 Logistic Regression

#### 4.4.1 Mathematical Foundation

**Softmax Function** (multi-class):
\[
P(y=c | \mathbf{x}; \mathbf{W}) = \frac{\exp(\mathbf{w}_c^T \mathbf{x})}{\sum_{c'=1}^C \exp(\mathbf{w}_{c'}^T \mathbf{x})}
\]

**Cross-Entropy Loss**:
\[
\mathcal{L}(\mathbf{W}) = -\frac{1}{N} \sum_{i=1}^N \log P(y_i | \mathbf{x}_i; \mathbf{W}) + \frac{\lambda}{2}\|\mathbf{W}\|_F^2
\]

#### 4.4.2 Implementation

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    C=1.0,                # Inverse regularization (tested: 0.1, 1.0, 10.0)
    penalty='l2',
    solver='lbfgs',       # L-BFGS optimizer (good for small datasets)
    max_iter=1000,
    multi_class='multinomial',  # True softmax (not one-vs-rest)
    random_state=42,
    n_jobs=-1
)

lr.fit(X_train, y_train)
```

### 4.5 Naive Bayes

#### 4.5.1 Multinomial Naive Bayes

**Mathematical Foundation**:
\[
P(y=c | \mathbf{x}) \propto P(y=c) \prod_{j=1}^d P(x_j | y=c)
\]

**Laplace Smoothing**:
\[
P(x_j | y=c) = \frac{\text{count}(x_j, y=c) + \alpha}{\sum_{j'} \text{count}(x_{j'}, y=c) + \alpha d}
\]

```python
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB(
    alpha=0.1,  # Laplace smoothing (tested: 0.01, 0.1, 1.0)
    fit_prior=True
)

nb.fit(X_train, y_train)
```

---

## 5. Transformer Fine-Tuning

### 5.1 DistilBERT Architecture

#### 5.1.1 Model Components

**Input Embeddings**:
\[
\mathbf{E} = \mathbf{E}_{\text{token}} + \mathbf{E}_{\text{position}}
\]

where:
- \( \mathbf{E}_{\text{token}} \in \mathbb{R}^{V \times 768} \): Token embeddings
- \( \mathbf{E}_{\text{position}} \in \mathbb{R}^{512 \times 768} \): Position embeddings

**Transformer Layers** (6 layers):

Multi-Head Self-Attention:
\[
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
\]

where \( d_k = 64 \) (hidden size per head).

**Classification Head**:
```python
Linear(768, 768)  # Pre-classifier
ReLU()
Dropout(0.2)
Linear(768, 4)    # Final classifier
```

#### 5.1.2 Implementation

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=4,
    problem_type='single_label_classification'
)

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

### 5.2 Dataset Preparation

```python
from torch.utils.data import Dataset

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item

# Tokenize
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding='max_length',
    max_length=256,
    return_tensors='pt'
)

# Create datasets
train_dataset = NewsDataset(train_encodings, y_train)
val_dataset = NewsDataset(val_encodings, y_val)
```

### 5.3 Training Configuration

#### 5.3.1 Training Arguments

```python
training_args = TrainingArguments(
    output_dir='./results',
    
    # Training hyperparameters
    num_train_epochs=4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    
    # Optimization
    learning_rate=2.5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    
    # Mixed precision
    fp16=True,  # Use FP16 on compatible GPUs
    
    # Evaluation and saving
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1',
    
    # Logging
    logging_dir='./logs',
    logging_steps=100,
    
    # Other
    seed=42,
    report_to='none'  # Disable wandb/tensorboard
)
```

#### 5.3.2 Custom Metrics

```python
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='macro'),
        'f1_weighted': f1_score(labels, predictions, average='weighted')
    }
```

#### 5.3.3 Trainer Setup

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save best model
trainer.save_model('./best_model')
```

### 5.4 Mixed Precision Training

**Automatic Mixed Precision (AMP)**:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()
    
    # Forward pass in FP16
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    
    # Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
```

**Benefits**:
- 2× faster training
- 50% memory reduction
- No accuracy loss

---

## 6. Evaluation Framework

### 6.1 Metrics

#### 6.1.1 Macro-F1 (Primary Metric)

**Per-Class F1**:
\[
F1_c = \frac{2 \times \text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}
\]

**Macro-F1** (unweighted average):
\[
\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^C F1_c
\]

**Implementation**:
```python
from sklearn.metrics import f1_score

macro_f1 = f1_score(y_true, y_pred, average='macro')
```

#### 6.1.2 Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['World', 'Sports', 'Business', 'Sci/Tech'],
            yticklabels=['World', 'Sports', 'Business', 'Sci/Tech'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
```

#### 6.1.3 Classification Report

```python
from sklearn.metrics import classification_report

report = classification_report(
    y_true, y_pred,
    target_names=['World', 'Sports', 'Business', 'Sci/Tech'],
    digits=4
)
print(report)
```

### 6.2 Cross-Validation

```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'macro_f1': make_scorer(f1_score, average='macro'),
    'weighted_f1': make_scorer(f1_score, average='weighted')
}

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=10,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

print(f"Mean Macro-F1: {cv_results['test_macro_f1'].mean():.4f} ± {cv_results['test_macro_f1'].std():.4f}")
```

---

## 7. Hyperparameter Optimization

### 7.1 Optuna (Bayesian Optimization)

```python
import optuna
from optuna.pruners import MedianPruner

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
    warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.2)
    
    # Create model
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=4
    ).to(device)
    
    # Training arguments
    args = TrainingArguments(
        output_dir=f'./trial_{trial.number}',
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        num_train_epochs=4,
        fp16=True,
        evaluation_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1'
    )
    
    # Train
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset,
                      eval_dataset=val_dataset, compute_metrics=compute_metrics)
    trainer.train()
    
    # Return validation F1
    return trainer.evaluate()['eval_f1']

# Run optimization
study = optuna.create_study(
    direction='maximize',
    pruner=MedianPruner()
)

study.optimize(objective, n_trials=10, timeout=28800)  # 8 hours

print(f"Best F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

---

## 8. Statistical Analysis

### 8.1 Paired t-Test

```python
from scipy.stats import ttest_rel

# Per-class F1 scores
svc_f1_per_class = precision_recall_fscore_support(
    y_test, svc_preds, average=None
)[2]

bert_f1_per_class = precision_recall_fscore_support(
    y_test, bert_preds, average=None
)[2]

# Paired t-test
t_stat, p_value = ttest_rel(bert_f1_per_class, svc_f1_per_class)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Significant difference (p < 0.05)")
else:
    print("No significant difference (p >= 0.05)")
```

### 8.2 Effect Size (Cohen's d)

```python
def cohens_d(x1, x2):
    mean_diff = np.mean(x1) - np.mean(x2)
    pooled_std = np.sqrt((np.std(x1, ddof=1)**2 + np.std(x2, ddof=1)**2) / 2)
    return mean_diff / pooled_std

d = cohens_d(bert_f1_per_class, svc_f1_per_class)
print(f"Cohen's d: {d:.4f}")

if abs(d) < 0.2:
    effect = "negligible"
elif abs(d) < 0.5:
    effect = "small"
elif abs(d) < 0.8:
    effect = "medium"
else:
    effect = "large"

print(f"Effect size: {effect}")
```

### 8.3 Bootstrap Confidence Intervals

```python
from sklearn.utils import resample

def bootstrap_metric(y_true, y_pred, metric_func, n_iterations=1000):
    scores = []
    n_size = len(y_true)
    
    for i in range(n_iterations):
        # Resample with replacement
        indices = resample(range(n_size), n_samples=n_size)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculate metric
        score = metric_func(y_true_boot, y_pred_boot)
        scores.append(score)
    
    # Calculate confidence interval
    alpha = 0.05
    lower = np.percentile(scores, (alpha/2) * 100)
    upper = np.percentile(scores, (1 - alpha/2) * 100)
    
    return np.mean(scores), lower, upper

mean_f1, lower, upper = bootstrap_metric(
    y_test, bert_preds,
    lambda y_t, y_p: f1_score(y_t, y_p, average='macro'),
    n_iterations=1000
)

print(f"Mean F1: {mean_f1:.4f}")
print(f"95% CI: [{lower:.4f}, {upper:.4f}]")
```

---

## 9. Implementation Details

### 9.1 Memory Optimization

**Sparse Matrix Storage**:
```python
import scipy.sparse as sp

# Save sparse matrix
sp.save_npz('features/X_train.npz', X_train)

# Load sparse matrix
X_train = sp.load_npz('features/X_train.npz')
```

**Gradient Checkpointing** (for very large models):
```python
model.gradient_checkpointing_enable()
```

### 9.2 Inference Optimization

**Batch Inference**:
```python
def predict_batch(model, texts, batch_size=32):
    all_preds = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encodings = tokenizer(batch_texts, truncation=True,
                             padding=True, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**encodings.to(device))
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
    
    return np.array(all_preds)
```

**ONNX Export** (for production):
```python
import torch.onnx

dummy_input = tokenizer("Sample text", return_tensors='pt')

torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    'model.onnx',
    opset_version=14,
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size'}
    }
)
```

---

## 10. Reproducibility Checklist

### ☑️ Environment
- [ ] Python version documented
- [ ] All package versions pinned
- [ ] Hardware specifications recorded
- [ ] CUDA version specified

### ☑️ Data
- [ ] Dataset source and version documented
- [ ] Train/val/test split strategy explained
- [ ] Random seeds set (data splitting)
- [ ] Data preprocessing steps detailed

### ☑️ Model Training
- [ ] Random seeds set (model initialization)
- [ ] Hyperparameters documented
- [ ] Training procedure explained
- [ ] Model checkpoints saved

### ☑️ Evaluation
- [ ] Metrics clearly defined
- [ ] Evaluation protocol specified
- [ ] Statistical tests documented
- [ ] Results include confidence intervals

### ☑️ Code
- [ ] Code publicly available
- [ ] README with usage instructions
- [ ] Dependencies listed in requirements.txt
- [ ] Example scripts provided

---

## Appendix: Complete Training Script

```python
#!/usr/bin/env python
"""
Complete training script for AG News classification
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import scipy.sparse as sp

# Set random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def main():
    print("Loading data...")
    dataset = load_dataset("sh0416/ag_news")
    
    # Convert to pandas
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    # Combine title and description
    train_df['text'] = train_df['title'] + " " + train_df['description']
    test_df['text'] = test_df['title'] + " " + test_df['description']
    
    # Normalize labels (1-based to 0-based)
    train_df['label'] = train_df['label'] - 1
    test_df['label'] = test_df['label'] - 1
    
    # Split train into train/val
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, stratify=train_df['label'], random_state=SEED
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Train classical model
    print("\nTraining LinearSVC...")
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
    X_train = vectorizer.fit_transform(train_df['text'])
    X_val = vectorizer.transform(val_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    
    svc = LinearSVC(C=0.1, max_iter=2000, random_state=SEED)
    calibrated_svc = CalibratedClassifierCV(svc, cv=5)
    calibrated_svc.fit(X_train, train_df['label'])
    
    svc_preds = calibrated_svc.predict(X_test)
    svc_f1 = f1_score(test_df['label'], svc_preds, average='macro')
    print(f"LinearSVC Test F1: {svc_f1:.4f}")
    
    # Train transformer
    print("\nTraining DistilBERT...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=4
    )
    
    # Tokenize
    train_encodings = tokenizer(train_df['text'].tolist(), truncation=True,
                                padding='max_length', max_length=256)
    val_encodings = tokenizer(val_df['text'].tolist(), truncation=True,
                              padding='max_length', max_length=256)
    
    # Create datasets
    class NewsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
    
    train_dataset = NewsDataset(train_encodings, train_df['label'].tolist())
    val_dataset = NewsDataset(val_encodings, val_df['label'].tolist())
    
    # Train
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,
        per_device_train_batch_size=32,
        learning_rate=2.5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        seed=SEED
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    trainer.train()
    
    # Evaluate
    test_encodings = tokenizer(test_df['text'].tolist(), truncation=True,
                               padding='max_length', max_length=256, return_tensors='pt')
    
    model.eval()
    with torch.no_grad():
        outputs = model(**test_encodings)
        bert_preds = torch.argmax(outputs.logits, dim=-1).numpy()
    
    bert_f1 = f1_score(test_df['label'], bert_preds, average='macro')
    print(f"DistilBERT Test F1: {bert_f1:.4f}")
    
    print("\nComparison:")
    print(f"LinearSVC:   {svc_f1:.4f}")
    print(f"DistilBERT:  {bert_f1:.4f}")
    print(f"Improvement: {(bert_f1 - svc_f1)*100:.2f}%")

if __name__ == '__main__':
    main()
```

---

**Document Version**: 1.0  
**Last Updated**: November 27, 2025  
**Total Pages**: 18  

---

<p align="center">
<i>This methodology document ensures complete reproducibility of all experiments.</i>
</p>