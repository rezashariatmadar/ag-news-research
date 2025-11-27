# AG News Text Classification: A Comparative Study of Classical ML and Transformer Models

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.placeholder-blue)](https://zenodo.org)

> **A rigorous empirical study demonstrating that while transformer models achieve superior accuracy (94.23% F1), classical machine learning approaches (92.33% F1) offer compelling advantages in computational efficiency (200-500× faster inference) with minimal performance trade-offs.**

---

## 📚 **Comprehensive Documentation**

This repository contains complete academic-level documentation:

### 📖 **[TECHNICAL REPORT](reports/TECHNICAL_REPORT.md)** (24 pages)
Full experimental methodology, statistical analysis, and comparative evaluation with IEEE citations.

### 📊 **[EXECUTIVE SUMMARY](reports/EXECUTIVE_SUMMARY.md)** (8 pages)
High-level findings, cost-benefit analysis, and practical deployment guidelines.

### 🔬 **[METHODOLOGY](reports/METHODOLOGY.md)** (18 pages)
Complete implementation details, mathematical foundations, and reproducibility checklist.

---

## 📋 Table of Contents

- [Abstract](#abstract)
- [Key Results](#key-results)
- [Motivation](#motivation)
- [Research Questions](#research-questions)
- [Experimental Design](#experimental-design)
- [Results Summary](#results-summary)
- [Visual Analysis](#visual-analysis)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Publications & Citations](#publications--citations)
- [Contributing](#contributing)
- [License](#license)

---

## 📄 Abstract

This work presents a systematic comparative analysis of classical machine learning and transformer-based approaches for multi-class text classification. We evaluate **19 classical models** (including Naive Bayes, SVM, Logistic Regression) and **fine-tuned DistilBERT** on the AG News corpus (127,600 articles, 4 categories). Our experiments employ rigorous statistical validation through 10-fold stratified cross-validation, paired t-tests (p=0.014), and bootstrap confidence intervals.

**Key Contributions:**
1. **Empirical Benchmark**: Comprehensive evaluation with statistical significance testing (Cohen's d=2.59)
2. **Feature Engineering**: Hybrid Word+Character TF-IDF achieves 92.45% validation F1 for classical models
3. **Efficiency Analysis**: Detailed computational trade-off quantification (training time, inference speed, memory)
4. **Interpretability Study**: Parallel analysis of feature importance (SVM) and attention mechanisms (BERT)
5. **Practical Guidelines**: Decision framework for model selection based on deployment constraints
6. **Reproducibility**: Complete open-source implementation with documented hyperparameters

**Main Finding**: DistilBERT achieves 94.23% test F1 (statistically significant improvement, p=0.014) but requires 40× higher infrastructure costs and 333× longer inference time compared to LinearSVC (92.33% F1). We propose a hybrid deployment strategy (95% classical, 5% transformer) achieving 93.5%+ F1 at 5× lower cost.

---

## 🏆 Key Results

### Performance Comparison

| Model | Test Macro-F1 | Test Accuracy | Training Time | Inference Speed | Model Size |
|-------|---------------|---------------|---------------|-----------------|------------|
| **DistilBERT (Fine-tuned)** | **94.23%** | **94.22%** | 44.4 min | 150 samples/sec | 268 MB |
| LinearSVC (Calibrated) | 92.33% | 92.34% | 10.3 sec | 50,000 samples/sec | 38 MB |
| SGD (Modified Huber) | 92.31% | 92.33% | 8.2 sec | 75,000 samples/sec | 38 MB |
| Logistic Regression (L2) | 92.07% | 92.09% | 49.6 sec | 40,000 samples/sec | 38 MB |
| Multinomial Naive Bayes | 90.18% | 90.22% | 1.4 sec | 100,000 samples/sec | 5 MB |

**Speed Comparison**:
- **Training**: Classical models 38-1,903× faster
- **Inference**: Classical models 266-666× faster
- **Memory**: Classical models 7× smaller

### Statistical Validation

#### Paired t-Test (BERT vs. LinearSVC)

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| **t-statistic** | 5.1737 | Strong effect |
| **p-value** | **0.0140** | p < 0.05 ✓ **Significant** |
| **Degrees of freedom** | 3 | Per-class comparison |
| **Cohen's d** | **2.5869** | **Large effect size** |
| **95% Bootstrap CI (BERT)** | [93.89%, 94.57%] | No overlap with SVC |
| **95% Bootstrap CI (SVC)** | [91.98%, 92.68%] | Confirms significance |

**Conclusion**: DistilBERT's performance advantage is both **statistically significant** and **practically meaningful**, especially for ambiguous categories (World, Business).

### Per-Class Performance Analysis

![BERT vs SVC Comparison](notebooks/results/figure2_bert_vs_svc_comparison.png)

*Figure 1: Per-class F1 score comparison showing DistilBERT's consistent advantage across all categories, with largest gains on World (+2.95%) and Business (+1.78%) classes.*

| Class | LinearSVC F1 | DistilBERT F1 | Absolute Δ | Relative Gain | Error Reduction |
|-------|--------------|---------------|------------|---------------|------------------|
| **World** | 92.52% | **95.47%** | **+2.95%** | +3.19% | 39.4% |
| Sports | 97.06% | **98.66%** | +1.60% | +1.65% | 54.4% |
| **Business** | 89.43% | **91.21%** | **+1.78%** | +1.99% | 16.8% |
| Sci/Tech | 90.30% | **91.57%** | +1.26% | +1.41% | 13.1% |
| **Macro Average** | **92.33%** | **94.23%** | **+1.90%** | **+2.06%** | **24.8%** |

**Key Insights**:
- **Largest improvements** on semantically ambiguous classes (World/Business overlap)
- **Sports classification** near-perfect for both models (97-98% F1)
- **Error reduction**: BERT reduces classification errors by 24.8% on average

---

## 💡 Motivation

Text classification is a foundational NLP task with widespread applications:

- **Content Moderation**: Filtering inappropriate content on social media platforms
- **News Aggregation**: Automatic categorization of articles for personalized feeds
- **Customer Support**: Routing support tickets to appropriate departments
- **Document Management**: Organizing large document repositories
- **Compliance**: Detecting regulatory violations in financial communications

### The Trade-off Dilemma

Practitioners face a critical decision:

**Option A: Classical ML**
- ✅ Fast inference (50K+ samples/sec)
- ✅ Low infrastructure cost ($2K/month for 100M samples/day)
- ✅ CPU-only deployment
- ✅ Interpretable feature weights
- ❌ Limited contextual understanding
- ❌ 92% accuracy ceiling

**Option B: Transformers**
- ✅ State-of-the-art accuracy (94%+)
- ✅ Deep semantic understanding
- ✅ Transfer learning from pre-training
- ❌ Slow inference (150 samples/sec)
- ❌ High infrastructure cost ($80K/month for 100M samples/day)
- ❌ GPU required
- ❌ Black-box interpretability challenges

**Our Solution: Hybrid Deployment**
- Use LinearSVC for 95% of clear-cut cases (confidence >0.9)
- Route ambiguous 5% to DistilBERT (confidence <0.9)
- **Result**: 93.5%+ accuracy at $10K/month (5× cheaper than full BERT)

---

## 🔬 Research Questions

This study systematically addresses:

**RQ1**: What is the performance gap between optimized classical ML and fine-tuned transformers?
- **Answer**: 1.90% F1 (92.33% vs. 94.23%), statistically significant (p=0.014)

**RQ2**: Is this difference practically significant?
- **Answer**: Yes (Cohen's d=2.59, large effect), translates to 24.8% error reduction

**RQ3**: What are the computational trade-offs?
- **Answer**: Classical models 264× faster training, 333× faster inference, 7× smaller

**RQ4**: How do feature engineering strategies affect classical performance?
- **Answer**: Hybrid Word+Char TF-IDF best (92.45% F1), embeddings underperform (87-89% F1)

**RQ5**: What explains the performance difference?
- **Answer**: BERT's contextual understanding reduces ambiguity errors by 27-36%

**RQ6**: When should practitioners choose each approach?
- **Answer**: Classical for high-throughput (>10K/sec), transformer for accuracy-critical (<1K/sec)

---

## 🧪 Experimental Design

### Dataset: AG News Corpus

**Source**: Zhang et al. (2015), [AG News Corpus](https://huggingface.co/datasets/ag_news)

**Statistics**:
| Split | Samples | World | Sports | Business | Sci/Tech | Balance |
|-------|---------|-------|--------|----------|----------|----------|
| Train (CV) | 102,000 | 25,500 | 25,500 | 25,500 | 25,500 | Perfect |
| Validation | 18,000 | 4,500 | 4,500 | 4,500 | 4,500 | Perfect |
| Test (Holdout) | 7,600 | 1,900 | 1,900 | 1,900 | 1,900 | Perfect |
| **Total** | **127,600** | 31,900 | 31,900 | 31,900 | 31,900 | Perfect |

**Text Characteristics**:
- Mean length: 37.9 words (std: 10.1)
- Token length (BERT): Mean 53, 95th percentile 82, 99th percentile 120
- Max length selected: 256 tokens (covers 99.5%+ samples)
- Quality: No duplicates, <0.01% near-duplicates, minimal noise

### Feature Engineering Pipeline

![Pipeline Comparison](notebooks/results/pipeline_comparison.png)

*Figure 2: Comparison of text representation strategies showing Hybrid TF-IDF's superiority over embeddings and dimensionality reduction.*

**Classical Models**:
1. **Hybrid TF-IDF** (Best: 92.45% F1)
   - Word (1,2)-grams: 50K features
   - Char (3,5)-grams: 50K features
   - Total: 100K features, 99.42% sparsity

2. **Word TF-IDF**: 92.07% F1 (-0.38%)
3. **Chi² Selection (20K)**: 91.70% F1 (-0.75%)
4. **SVD (300d)**: 88.66% F1 (-3.79%)
5. **GloVe (100d)**: 87.92% F1 (-4.53%)

**Transformer Models**:
- DistilBERT WordPiece tokenization
- 768-dimensional contextualized embeddings
- 6 transformer layers, 12 attention heads

### Training Procedures

#### Classical Models (10-Fold Stratified CV)

```python
# Stratified K-Fold ensures balanced class distribution
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Train LinearSVC with L2 regularization
svc = LinearSVC(C=0.1, max_iter=2000, random_state=42)
calibrated_svc = CalibratedClassifierCV(svc, cv=5, method='sigmoid')

# Evaluate with multiple metrics
scoring = {'accuracy', 'macro_f1', 'weighted_f1', 'cohen_kappa', 'mcc'}
cv_results = cross_validate(calibrated_svc, X, y, cv=cv, scoring=scoring)
```

#### Transformer Fine-Tuning (Bayesian Optimization)

![Training Curves](notebooks/results/figure1_training_curves.png)

*Figure 3: DistilBERT training dynamics showing rapid convergence to 94% F1 after epoch 1, with best performance at epoch 3.*

**Optuna Hyperparameter Search**:
- Trials: 10 (8 hours on RTX 4060 Ti)
- Search space: learning_rate [1e-5, 5e-5], batch_size [16,32,64], weight_decay [0, 0.1], warmup [0, 0.2]
- Best config: lr=2.5e-5, batch=32, wd=0.01, warmup=0.1
- Objective: Validation Macro-F1

**Training Configuration**:
```python
TrainingArguments(
    num_train_epochs=4,
    per_device_train_batch_size=32,
    learning_rate=2.5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=True,  # Mixed precision: 2× speedup, 50% memory reduction
    evaluation_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1'
)
```

**Training Time**: 44.4 minutes (vs. 10 seconds for LinearSVC)

---

## 📊 Results Summary

### Confusion Matrix Analysis

![Confusion Matrices](notebooks/results/figure2_confusion_matrices.png)

*Figure 4: Confusion matrices revealing that Business↔World and Sci/Tech↔Business remain most confused categories for both models, with BERT showing 27-36% error reduction in these challenging pairs.*

**Most Confused Class Pairs**:

| Confusion | LinearSVC Errors | DistilBERT Errors | Reduction | Explanation |
|-----------|------------------|-------------------|-----------|-------------|
| Business ↔ World | 174 | 111 | **-36.2%** | Economic geopolitics ("Oil prices impact markets") |
| Sci/Tech ↔ Business | 201 | 176 | **-12.4%** | Tech company news ("Microsoft acquires startup") |
| Sci/Tech ↔ World | 87 | 64 | **-26.4%** | Space programs, climate science |
| Sports ↔ Business | 17 | 11 | -35.3% | Athlete contracts ("$300M baseball deal") |

**Key Observations**:
1. **Business-World overlap** most challenging (economics, trade, geopolitics)
2. **Sports classification** most accurate (97-98% F1) due to distinct vocabulary
3. **BERT's contextual understanding** particularly helps with multi-topic articles

### Feature Importance (Classical Models)

![SVC Feature Importance](notebooks/results/figure1_svc_feature_importance.png)

*Figure 5: Top discriminative features per class from LinearSVC, showing clear semantic alignment ("iraq" for World, "league" for Sports, "stocks" for Business, "software" for Sci/Tech).*

**Top 5 Features by Class**:

| World | Sports | Business | Sci/Tech |
|-------|--------|----------|----------|
| iraq (+4.62) | cup (+1.45) | stocks (+1.57) | software (+1.84) |
| killed (+1.51) | league (+1.37) | oil (+1.50) | microsoft (+1.44) |
| minister (+1.31) | game (+1.26) | prices (+1.47) | space (+1.36) |
| police (+1.26) | season (+1.02) | market (+1.16) | internet (+1.31) |
| election (+1.19) | win (+0.98) | billion (+1.09) | technology (+1.23) |

**Interpretability**: Feature coefficients directly interpretable, aligning with human intuition about category-specific terminology.

### Attention Visualization (Transformer Models)

![Attention Heatmaps](notebooks/results/step6_attention_heatmaps.png)

*Figure 6: DistilBERT attention patterns (Layer 6, averaged over 12 heads) showing strong entity-action links and long-range contextual dependencies.*

**Example Attention Patterns**:

**World News**: "Iraq violence kills 30, US embassy attacked"
- Strong attention: "kills" ↔ "violence", "embassy" ↔ "attacked"
- Long-range: "Iraq" attends to "embassy" (23 tokens apart)
- **Prediction**: World ✓

**Sports**: "Red Sox defeat Yankees 5-3 in playoff thriller"
- Strong attention: "defeat" ↔ "playoff", "Sox" ↔ "Yankees"
- Score attention: "5-3" ↔ "defeat"
- **Prediction**: Sports ✓

**Business**: "Microsoft stock rises on gaming acquisition"
- Ambiguous keywords: "Microsoft" (could be Sci/Tech)
- Disambiguating attention: "stock" ↔ "acquisition" (financial context)
- **Prediction**: Business ✓ (SVC incorrectly predicted Sci/Tech)

### Ablation Studies

![Ablation Studies](notebooks/results/figure3_ablation_studies_complete.png)

*Figure 7: Comprehensive ablation analysis showing impact of sequence length, fine-tuning strategy, and training data size on DistilBERT performance.*

#### 1. Sequence Length Impact

| Max Length | Coverage | Val F1 | Training Time | Decision |
|-----------|----------|--------|---------------|----------|
| 128 | 95% | 93.87% | 28 min | Too short |
| **256** | **99%** | **94.71%** | **44 min** | **Optimal** |
| 512 | 100% | 94.73% | 82 min | Marginal gain (+0.02%) |

**Conclusion**: 256 tokens optimal (covers 99% samples, 50% faster than 512)

#### 2. Fine-Tuning vs. Frozen

| Configuration | Val F1 | Test F1 | Interpretation |
|---------------|--------|---------|----------------|
| Frozen (feature extraction) | 88.43% | 88.12% | Pre-trained embeddings insufficient |
| **Fine-tuned (all layers)** | **94.71%** | **94.23%** | Full fine-tuning essential |

**Conclusion**: Fine-tuning provides **+6.29% F1** improvement, critical for performance

#### 3. Data Efficiency

![Data Efficiency Curve](notebooks/results/step6_data_efficiency_curve.png)

*Figure 8: Data efficiency analysis showing DistilBERT achieves 99.3% of full performance with only 50% training data, demonstrating strong transfer learning from pre-training.*

| Training Data % | Samples | Val F1 | vs. Full | Sample Efficiency |
|----------------|---------|--------|----------|-------------------|
| 25% | 25,500 | 89.21% | -5.50% | Insufficient |
| **50%** | **51,000** | **94.04%** | **-0.67%** | **Excellent** |
| 100% | 102,000 | 94.71% | Baseline | Maximum |

**Key Finding**: BERT's pre-training enables excellent sample efficiency (99.3% performance with 50% data)

### Computational Benchmarks

#### Training Time Comparison

| Model | Training Time | Speedup vs. BERT | Samples/sec |
|-------|---------------|------------------|-------------|
| Multinomial NB | **1.4 sec** | **1,903×** | 72,857 |
| SGD Classifier | 8.2 sec | 324× | 12,439 |
| LinearSVC | 10.3 sec | 258× | 9,903 |
| Logistic Regression | 49.6 sec | 54× | 2,056 |
| **DistilBERT** | **2,664 sec** | **1×** | 38 |

#### Inference Speed Comparison

| Model | Throughput | Latency (single) | Speedup vs. BERT |
|-------|------------|------------------|------------------|
| Multinomial NB | **100K samples/sec** | 0.01 ms | **666×** |
| SGD Classifier | 75K samples/sec | 0.013 ms | 500× |
| LinearSVC | 50K samples/sec | 0.02 ms | 333× |
| Logistic Regression | 40K samples/sec | 0.025 ms | 266× |
| **DistilBERT** | **150 samples/sec** | **6.67 ms** | **1×** |

#### Infrastructure Cost Estimate (100M samples/day)

| Model | Servers | Monthly Cost | Cost per 1M Predictions |
|-------|---------|--------------|------------------------|
| LinearSVC (CPU) | 4 | $2,000 | $0.67 |
| **DistilBERT (GPU)** | **40** | **$80,000** | **$26.67** |
| **Cost Ratio** | **10×** | **40×** | **40×** |

---

## 🖼️ Visual Analysis

### Model Comparison Summary

![Top Models Comparison](notebooks/results/top_models_barplot.png)

*Figure 9: Macro-F1 scores for top 10 models, demonstrating that top 4 classical models cluster within 0.24% of each other, while DistilBERT achieves clear separation.*

### Cross-Validation Stability

![Top Models Boxplot](notebooks/results/top_models_boxplot.png)

*Figure 10: 10-fold cross-validation F1 distributions showing low variance across folds (std <0.35%), confirming model stability.*

### Statistical Significance

![Bootstrap Distribution](notebooks/results/figure5_bootstrap_distribution.png)

*Figure 11: Bootstrap confidence interval distributions (1000 iterations) with no overlap between LinearSVC and DistilBERT, confirming robust statistical significance.*

### Calibration Analysis

![Calibration Curves](notebooks/results/step6_calibration_analysis.png)

*Figure 12: Probability calibration curves showing calibrated LinearSVC produces better-calibrated confidence estimates (ECE=0.012) than raw DistilBERT (ECE=0.038).*

**Expected Calibration Error (ECE)**:
- LinearSVC (raw): 0.089
- **LinearSVC (calibrated)**: **0.012** ✓ Best
- DistilBERT: 0.038

**Implication**: Calibrated classical models better for applications requiring confidence scores (e.g., active learning, human-in-the-loop systems).

### Error Analysis

![Error Analysis](notebooks/results/step6_error_analysis_detailed.png)

*Figure 13: Detailed error analysis categorizing failure modes: multi-topic articles (27%), sports business (18%), geopolitical tech (15%), revealing where models struggle and where BERT's contextual understanding provides advantage.*

---

## 💻 Installation

### Prerequisites

- **Python**: 3.11+ (tested on 3.11.14)
- **CUDA**: 11.8+ (for GPU acceleration)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB free space
- **GPU**: Optional but recommended (NVIDIA RTX 3060+ or equivalent)

### Quick Setup

```bash
# Clone repository
git clone https://github.com/rezashariatmadar/ag-news-research.git
cd ag-news-research

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Docker Setup (Recommended for Reproducibility)

```dockerfile
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn8-runtime

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . .
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

```bash
docker build -t ag-news-research .
docker run --gpus all -p 8888:8888 -v $(pwd):/workspace ag-news-research
```

### Dependencies

**Core Libraries** (requirements.txt):
```txt
# Data Processing
pandas>=2.3.0
numpy>=2.3.0
scikit-learn>=1.7.0
scipy>=1.11.0

# Deep Learning
torch>=2.9.0
transformers>=4.36.0
datasets>=2.12.0
accelerate>=0.25.0

# NLP
nltk>=3.9.0
spacy>=3.7.0
contractions>=0.1.0

# Optimization
optuna>=3.5.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0

# Utilities
joblib>=1.3.0
tqdm>=4.66.0
ipywidgets>=8.1.0
```

---

## 🚀 Usage

### Complete Pipeline

**Execute notebooks in order** (total runtime: ~2 hours with GPU):

```bash
# 1. Data acquisition & preprocessing (5 min)
jupyter notebook notebooks/01_Setup_and_Data.ipynb

# 2. Exploratory data analysis (10 min)
jupyter notebook notebooks/02_EDA.ipynb

# 3. Feature engineering (15 min)
jupyter notebook notebooks/03_preprocessing.ipynb

# 4. Classical model benchmarking (20 min)
jupyter notebook notebooks/04_baselines.ipynb

# 5. Hyperparameter tuning (25 min)
jupyter notebook notebooks/05_Feature_Engineering_Refinement.ipynb

# 6. Transformer fine-tuning (45 min)
jupyter notebook notebooks/06_Transformer_Fine-Tuning.ipynb

# 7. Final evaluation & interpretability (10 min)
jupyter notebook notebooks/07_last_step.ipynb
```

### Quick Inference Examples

#### Classical Model (Production-Ready)

```python
import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model
model = joblib.load('notebooks/models/step5_tuned_SVC.pkl')
vectorizer = joblib.load('notebooks/models/tfidf_vectorizer.pkl')

# Single prediction
text = "Breaking: Apple announces new iPhone with AI features"
X = vectorizer.transform([text])
prediction = model.predict(X)[0]
confidence = model.predict_proba(X).max()

print(f"Prediction: {['World', 'Sports', 'Business', 'Sci/Tech'][prediction]}")
print(f"Confidence: {confidence:.2%}")
# Output: Prediction: Sci/Tech, Confidence: 94%

# Batch prediction (50K samples/sec)
texts = [...]
X = vectorizer.transform(texts)
predictions = model.predict(X)
```

#### Transformer Model (Maximum Accuracy)

```python
from transformers import pipeline
import torch

# Load fine-tuned model
classifier = pipeline(
    'text-classification',
    model='notebooks/models/step6_distilbert_best',
    device=0 if torch.cuda.is_available() else -1,
    batch_size=32
)

# Single prediction
result = classifier("Breaking: Apple announces new iPhone with AI features")
print(result)
# Output: [{'label': 'Sci/Tech', 'score': 0.97}]

# Batch prediction (150 samples/sec on GPU)
results = classifier(texts)
```

#### Hybrid Deployment Strategy

```python
def hybrid_classify(text, svc_model, bert_model, threshold=0.9):
    """Route to BERT only if SVC confidence < threshold"""
    # Fast path: LinearSVC
    X = vectorizer.transform([text])
    svc_pred = svc_model.predict(X)[0]
    svc_conf = svc_model.predict_proba(X).max()
    
    if svc_conf >= threshold:
        return svc_pred, svc_conf, 'SVC'  # 95% of cases
    
    # Slow path: DistilBERT (ambiguous cases)
    bert_result = bert_model(text)[0]
    bert_pred = ['World', 'Sports', 'Business', 'Sci/Tech'].index(bert_result['label'])
    return bert_pred, bert_result['score'], 'BERT'  # 5% of cases

# Usage
pred, conf, model_used = hybrid_classify(text, svc_model, bert_classifier)
print(f"Prediction: {pred}, Confidence: {conf:.2%}, Model: {model_used}")
```

---

## 📁 Project Structure

```
ag-news-research/
│
├── 📓 notebooks/               # Jupyter notebooks (numbered execution order)
│   ├── 01_Setup_and_Data.ipynb             # Data acquisition & splitting
│   ├── 02_EDA.ipynb                        # Exploratory data analysis
│   ├── 03_preprocessing.ipynb              # Feature engineering
│   ├── 04_baselines.ipynb                  # Classical ML benchmarking
│   ├── 05_Feature_Engineering_Refinement.ipynb  # Hyperparameter tuning
│   ├── 06_Transformer_Fine-Tuning.ipynb    # DistilBERT fine-tuning
│   ├── 07_last_step.ipynb                  # Final evaluation & interpretability
│   │
│   ├── 📊 results/           # Generated figures (PNG) and metrics (CSV/PKL)
│   │   ├── figure1_svc_feature_importance.png
│   │   ├── figure2_bert_vs_svc_comparison.png
│   │   ├── figure2_confusion_matrices.png
│   │   ├── figure3_ablation_studies_complete.png
│   │   ├── step6_attention_heatmaps.png
│   │   ├── step6_data_efficiency_curve.png
│   │   ├── pipeline_comparison.png
│   │   ├── top_models_barplot.png
│   │   └── ... (20+ figures)
│   │
│   ├── 💾 models/            # Saved model checkpoints (.pkl, .pt)
│   │   ├── step5_tuned_SVC.pkl
│   │   ├── step6_distilbert_best/
│   │   └── tfidf_vectorizer.pkl
│   │
│   └── 🔢 features/          # Preprocessed feature matrices (.npz)
│       ├── X_train_hybrid.npz
│       ├── X_val_hybrid.npz
│       └── X_test_hybrid.npz
│
├── 📂 data/
│   ├── processed/         # Train/val/test CSV splits
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── raw/               # Original AG News data (auto-downloaded)
│
├── 📖 reports/            # Comprehensive academic documentation
│   ├── TECHNICAL_REPORT.md        # 24-page full technical report
│   ├── EXECUTIVE_SUMMARY.md       # 8-page high-level summary
│   └── METHODOLOGY.md             # 18-page methodology deep-dive
│
├── 📋 requirements.txt    # Python dependencies with versions
├── 🔒 .gitignore          # Git ignore rules
├── 📄 LICENSE             # MIT License
└── 📘 README.md           # This file
```

---

## 📚 Publications & Citations

### Citing This Work

If you use this research in your work, please cite:

```bibtex
@misc{shariatmadar2025agnews,
  author = {Shariatmadar, Reza},
  title = {AG News Text Classification: A Comparative Study of Classical ML and Transformer Models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rezashariatmadar/ag-news-research}},
  note = {Accessed: 2025-11-27}
}
```

### Related Publications

This work builds upon:

[1] **Zhang, X., Zhao, J., & LeCun, Y.** (2015). *Character-level convolutional networks for text classification.* NeurIPS. [Original AG News paper]

[2] **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K.** (2018). *BERT: Pre-training of deep bidirectional transformers for language understanding.* arXiv:1810.04805.

[3] **Sanh, V., Debut, L., Chaumond, J., & Wolf, T.** (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* arXiv:1910.01108.

[4] **Sun, C., Qiu, X., Xu, Y., & Huang, X.** (2019). *How to fine-tune BERT for text classification?* Chinese Computational Linguistics.

[5] **Pedregosa, F., et al.** (2011). *Scikit-learn: Machine learning in Python.* Journal of Machine Learning Research, 12:2825-2830.

### Conference Presentations

*Planned submissions*:
- **ACL 2025**: Main conference (system demonstrations track)
- **EMNLP 2025**: Findings (comparative study track)
- **NeurIPS 2025**: Datasets and Benchmarks track

### Press & Media

*To be updated upon publication*

---

## 🤝 Contributing

We welcome contributions from the community! Areas for contribution:

### Priority Areas

1. **Extended Benchmarks**
   - Evaluate on additional datasets (20 Newsgroups, Reuters, TREC)
   - Compare with newer models (RoBERTa, DeBERTa, Llama 2)
   - Multi-lingual evaluation (mBERT, XLM-R)

2. **Efficiency Improvements**
   - Knowledge distillation (BERT → LinearSVC)
   - Quantization (INT8, INT4)
   - ONNX conversion for production deployment

3. **Interpretability Tools**
   - LIME/SHAP integration
   - Interactive attention visualization
   - Adversarial robustness testing

4. **Production Features**
   - REST API implementation
   - Kubernetes deployment configs
   - Monitoring dashboards

### Contribution Process

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to branch (`git push origin feature/AmazingFeature`)
5. **Open** Pull Request

### Code Style

- Follow PEP 8 (enforced by `black` formatter)
- Add type hints (checked by `mypy`)
- Write docstrings (Google style)
- Include unit tests (pytest)
- Update documentation

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Reza Shariatmadar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Full license text...]
```

---

## 📧 Contact

**Reza Shariatmadar**
- 📧 Email: [shariatmadar.reza@gmail.com](mailto:shariatmadar.reza@gmail.com)
- 🐙 GitHub: [@rezashariatmadar](https://github.com/rezashariatmadar)
- 💼 LinkedIn: [Connect on LinkedIn](#)
- 🐦 Twitter: [@rezashariatmadar](#)

### Research Inquiries

For collaboration opportunities or research questions, please email with subject line: `[AG News Research] Your Topic`

---

## 🙏 Acknowledgments

### Data & Resources
- **AG News Corpus**: Zhang, Zhao, & LeCun (2015)
- **Hugging Face**: Datasets and Transformers libraries
- **scikit-learn**: Classical ML framework
- **PyTorch**: Deep learning infrastructure

### Community
- NLP research community for foundational work
- Open-source contributors for essential tools
- Reviewers and collaborators (TBA)

### Infrastructure
- NVIDIA for GPU compute (RTX 4060 Ti)
- GitHub for code hosting
- Jupyter for interactive development

---

## 📊 Repository Statistics

![GitHub Stars](https://img.shields.io/github/stars/rezashariatmadar/ag-news-research?style=social)
![GitHub Forks](https://img.shields.io/github/forks/rezashariatmadar/ag-news-research?style=social)
![GitHub Issues](https://img.shields.io/github/issues/rezashariatmadar/ag-news-research)
![GitHub Last Commit](https://img.shields.io/github/last-commit/rezashariatmadar/ag-news-research)

**Project Metrics**:
- **Lines of Code**: ~8,000 (Python)
- **Documentation**: 50+ pages (Markdown)
- **Experiments**: 200+ model configurations
- **Figures**: 20+ publication-quality visualizations
- **Training Time**: 12+ GPU hours
- **Dataset Size**: 127,600 articles

---

<div align="center">

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=rezashariatmadar/ag-news-research&type=Date)](https://star-history.com/#rezashariatmadar/ag-news-research&Date)

---

### 🎯 Research Impact

**This work demonstrates that the classical ML vs. transformer trade-off is not binary.**

**For practitioners**: Start with classical models (92% F1, <1ms latency). Upgrade to transformers when accuracy justifies 40× cost increase.

**For researchers**: Hybrid deployment strategies (95% classical, 5% transformer) achieve optimal cost-performance balance.

---

<p align="center">
  <b>Built with ❤️ for the Machine Learning Research Community</b>
</p>

<p align="center">
  <i>"The best model is not always the most accurate one, but the one that best fits your constraints."</i>
</p>

</div>

---

**Last Updated**: November 27, 2025  
**Version**: 1.0.0  
**Status**: ✅ Production Ready