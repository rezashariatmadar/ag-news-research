# AG News Text Classification: A Comparative Study of Classical ML and Transformer Models

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Detailed Reports](#detailed-reports)
- [Results Visualization](#results-visualization)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

This research project presents a comprehensive comparative analysis of **classical machine learning** and **transformer-based models** for multi-class text classification on the AG News dataset. Our work systematically evaluates the performance, interpretability, and computational efficiency of various approaches, from traditional TF-IDF with linear classifiers to state-of-the-art DistilBERT transformers.

### Research Objectives

1. **Systematic Evaluation**: Benchmark 19+ classical ML models and transformer architectures
2. **Feature Engineering**: Develop and compare multiple text representation strategies
3. **Statistical Rigor**: Employ paired t-tests and bootstrap confidence intervals
4. **Model Interpretability**: Analyze feature importance and attention mechanisms
5. **Practical Guidelines**: Provide actionable insights for real-world deployment

### Dataset: AG News

- **Classes**: 4 (World, Sports, Business, Sci/Tech)
- **Training**: 120,000 samples
- **Validation**: 18,000 samples
- **Test**: 7,600 samples
- **Balance**: Perfectly balanced across all classes
- **Source**: [AG News Corpus](https://huggingface.co/datasets/ag_news)

---

## 🏆 Key Results

### Performance Comparison

| Model | Test Macro-F1 | Test Accuracy | Training Time | Inference Speed |
|-------|---------------|---------------|---------------|------------------|
| **DistilBERT (Fine-tuned)** | **94.23%** | **94.22%** | ~45 min | ~150 samples/sec |
| LinearSVC (Calibrated) | 92.33% | 92.34% | ~2 min | ~50,000 samples/sec |
| SGD (Modified Huber) | 92.31% | 92.33% | ~30 sec | ~75,000 samples/sec |
| Logistic Regression (L2) | 92.07% | 92.09% | ~1.5 min | ~40,000 samples/sec |
| Multinomial Naive Bayes | 90.18% | 90.22% | ~5 sec | ~100,000 samples/sec |

### Statistical Significance

- **Paired t-test**: DistilBERT vs. LinearSVC
  - **t-statistic**: 5.17
  - **p-value**: 0.014 (p < 0.05) ✓
  - **Cohen's d**: 2.59 (large effect)
  - **Conclusion**: DistilBERT significantly outperforms classical models

### Per-Class Performance (Test Set)

![Per-Class F1 Comparison](notebooks/results/figure2_bert_vs_svc_comparison.png)

| Class | LinearSVC F1 | DistilBERT F1 | Improvement |
|-------|--------------|---------------|-------------|
| World | 92.52% | **95.47%** | +2.95% |
| Sports | 97.06% | **98.66%** | +1.60% |
| Business | 89.43% | **91.21%** | +1.78% |
| Sci/Tech | 90.30% | **91.57%** | +1.26% |

### Key Findings

1. **🚀 Transformers Win on Accuracy**: DistilBERT achieves ~2% absolute improvement over best classical model
2. **⚡ Classical Models Win on Speed**: 200-500x faster inference, 90x faster training
3. **💾 Resource Efficiency**: Classical models require <100MB vs. ~250MB for DistilBERT
4. **📊 Data Efficiency**: DistilBERT reaches 99.3% of full performance with only 50% training data
5. **🎯 Hybrid Opportunity**: Ensemble of BERT + LinearSVC achieves marginal gains (94.31%)

---

## 📁 Project Structure

```
ag-news-research/
│
├── notebooks/              # Jupyter notebooks (execution order)
│   ├── 01_Setup_and_Data.ipynb            # Data acquisition & splitting
│   ├── 02_EDA.ipynb                       # Exploratory data analysis
│   ├── 03_preprocessing.ipynb             # Feature engineering
│   ├── 04_baselines.ipynb                 # Classical ML evaluation
│   ├── 05_Feature_Engineering_Refinement.ipynb  # Hyperparameter tuning
│   ├── 06_Transformer_Fine-Tuning.ipynb   # DistilBERT fine-tuning
│   ├── 07_last_step.ipynb                 # Final evaluation & interpretability
│   │
│   ├── results/           # Generated figures and metrics
│   ├── models/            # Saved model checkpoints
│   └── features/          # Preprocessed feature matrices
│
├── data/
│   ├── processed/         # Train/val/test splits
│   └── raw/               # Original AG News data (auto-downloaded)
│
├── reports/               # Detailed academic reports
│   ├── TECHNICAL_REPORT.md       # Full technical documentation
│   ├── EXECUTIVE_SUMMARY.md      # High-level findings
│   └── METHODOLOGY.md            # Detailed methodology
│
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md             # This file
```

---

## 🔬 Methodology

### 1. Data Preprocessing

**Text Cleaning Strategies**:
- Whitespace normalization
- URL/email tokenization vs. removal
- Contraction expansion
- Case normalization

**Tokenization Methods**:
- Whitespace splitting
- NLTK word tokenizer
- spaCy tokenizer
- Regex-based tokenization

### 2. Feature Engineering

#### Classical ML Features

**TF-IDF Vectorization**:
- **Word n-grams**: (1,2), (1,3) with 50K max features
- **Character n-grams**: (3,5), (3,6) with 50K max features
- **Hybrid**: Word + Char TF-IDF (100K features) → **Best performer**

**Feature Selection**:
- Chi-square (χ²): Top 20K features
- Mutual Information (MI)

**Dimensionality Reduction**:
- Truncated SVD (LSA): 100-500 components
- NMF: 50-150 components

**Embeddings**:
- GloVe 100d (pre-trained)
- Word2Vec (custom trained)
- TF-IDF weighted averaging

### 3. Model Training

#### Classical Models (10-Fold Stratified CV)

1. **Naive Bayes Family**
   - Multinomial NB (α=0.1, 1.0)
   - Complement NB (α=0.1, 1.0)
   - Bernoulli NB

2. **Linear Classifiers**
   - Logistic Regression (L2: C=0.1, 1.0, 10.0)
   - LinearSVC (C=0.1, 1.0, 10.0)
   - SGD Classifier (hinge, log, modified_huber)

3. **Tree-Based**
   - Random Forest (n=100, 200)

4. **Ensemble Methods**
   - Stacking (LR meta-learner)
   - Soft/Hard Voting

#### Transformer Models

**DistilBERT Fine-Tuning**:
- Base Model: `distilbert-base-uncased`
- Max Sequence Length: 256 tokens
- Batch Size: 32 (with gradient accumulation)
- Learning Rate: 2.5e-5 (optimized via Optuna)
- Warmup Ratio: 0.1
- Weight Decay: 0.01
- Mixed Precision (FP16): Enabled
- Early Stopping: Patience=2
- Optimizer: AdamW
- Scheduler: Linear with warmup

**Hyperparameter Optimization**:
- Framework: Optuna (Bayesian optimization)
- Trials: 10
- Pruning: Median pruner
- Metric: Validation Macro-F1

### 4. Evaluation Protocol

- **Cross-Validation**: 10-fold stratified for classical models
- **Metrics**: Macro-F1 (primary), Accuracy, Weighted-F1, Cohen's κ, MCC
- **Statistical Tests**: Paired t-test, Bootstrap CI (95%, 1000 samples)
- **Holdout Test**: Final evaluation on 7,600 unseen samples

### 5. Interpretability Analysis

#### Classical Models
- **Feature Importance**: Top-k SVC coefficients per class
- **Confusion Matrices**: Error pattern analysis

#### Transformers
- **Attention Visualization**: Layer-6 attention heatmaps
- **Error Analysis**: Manual inspection of misclassifications
- **Calibration**: Expected Calibration Error (ECE)

---

## 💻 Installation

### Prerequisites

- Python 3.11+
- CUDA 11.8+ (for GPU acceleration)
- 16GB RAM minimum
- 10GB disk space

### Setup

```bash
# Clone repository
git clone https://github.com/rezashariatmadar/ag-news-research.git
cd ag-news-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Requirements

```txt
# Core
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Deep Learning
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0

# NLP
nltk>=3.8
spacy>=3.5.0
sentence-transformers>=2.2.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Optimization
optuna>=3.1.0

# Utilities
joblib>=1.2.0
tqdm>=4.65.0
```

---

## 🚀 Usage

### Quick Start: Full Pipeline

```bash
# Execute notebooks in order
jupyter notebook notebooks/01_Setup_and_Data.ipynb
jupyter notebook notebooks/02_EDA.ipynb
jupyter notebook notebooks/03_preprocessing.ipynb
jupyter notebook notebooks/04_baselines.ipynb
jupyter notebook notebooks/05_Feature_Engineering_Refinement.ipynb
jupyter notebook notebooks/06_Transformer_Fine-Tuning.ipynb
jupyter notebook notebooks/07_last_step.ipynb
```

### Individual Components

#### Train Classical Model

```python
import joblib
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import scipy.sparse as sp

# Load preprocessed features
X_train = sp.load_npz('notebooks/features/X_train_hybrid.npz')
X_test = sp.load_npz('notebooks/features/X_test_hybrid.npz')
y_train = pd.read_csv('data/processed/train.csv')['label'].values
y_test = pd.read_csv('data/processed/test.csv')['label'].values

# Train LinearSVC with calibration
svc = LinearSVC(C=0.1, max_iter=2000, random_state=42)
calibrated_svc = CalibratedClassifierCV(svc, cv=5, method='sigmoid')
calibrated_svc.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import f1_score
preds = calibrated_svc.predict(X_test)
print(f"Test Macro-F1: {f1_score(y_test, preds, average='macro'):.4f}")

# Save model
joblib.dump(calibrated_svc, 'notebooks/models/calibrated_svc.pkl')
```

#### Fine-Tune DistilBERT

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=4
)

# Tokenize data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=32,
    learning_rate=2.5e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,  # Mixed precision
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1',
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()
```

#### Inference

```python
# Classical model
model = joblib.load('notebooks/models/step5_tuned_SVC.pkl')
X_new = vectorizer.transform(["Breaking news from Washington"])
prediction = model.predict(X_new)

# Transformer model
from transformers import pipeline
classifier = pipeline('text-classification', model='notebooks/models/step6_distilbert_best')
result = classifier("Breaking news from Washington")
print(result)
```

---

## 📊 Detailed Reports

For in-depth analysis, please refer to our comprehensive reports:

1. **[Technical Report](reports/TECHNICAL_REPORT.md)** (~20 pages)
   - Complete experimental setup
   - Detailed results and ablation studies
   - Statistical analysis
   - Failure case analysis

2. **[Executive Summary](reports/EXECUTIVE_SUMMARY.md)** (~5 pages)
   - High-level findings
   - Business implications
   - Deployment recommendations

3. **[Methodology Document](reports/METHODOLOGY.md)** (~15 pages)
   - Preprocessing pipeline
   - Feature engineering techniques
   - Model architectures
   - Training procedures

---

## 📈 Results Visualization

### Training Curves

![Training Curves](notebooks/results/figure1_training_curves.png)

### Confusion Matrices

![Confusion Matrix Comparison](notebooks/results/figure2_confusion_matrices.png)

### Attention Heatmaps (DistilBERT)

![Attention Heatmaps](notebooks/results/step6_attention_heatmaps.png)

### Feature Importance (LinearSVC)

![SVC Feature Importance](notebooks/results/figure1_svc_feature_importance.png)

### Data Efficiency

![Data Efficiency Curve](notebooks/results/step6_data_efficiency_curve.png)

---

## 🎓 Citation

If you use this work in your research, please cite:

```bibtex
@misc{shariatmadar2025agnews,
  author = {Shariatmadar, Reza},
  title = {AG News Text Classification: A Comparative Study of Classical ML and Transformer Models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rezashariatmadar/ag-news-research}}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Reza Shariatmadar**
- Email: shariatmadar.reza@gmail.com
- GitHub: [@rezashariatmadar](https://github.com/rezashariatmadar)

---

## 🙏 Acknowledgments

- AG News dataset creators
- HuggingFace team for Transformers library
- PyTorch team
- scikit-learn contributors

---

## 📚 References

[1] Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. *NeurIPS*.

[2] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*.

[3] Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.

[4] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of machine learning research*.

---

<p align="center">
  <i>Built with ❤️ for the ML research community</i>
</p>