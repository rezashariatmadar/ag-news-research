# AG News Dataset: Exploratory Data Analysis Report

## 1. Executive Summary
This report details the findings from the Deep Exploratory Data Analysis (EDA) conducted on the AG News dataset. The analysis covers document statistics, semantic manifold geometry, label noise detection, and robustness profiling.

**Key Insights:**
*   **Data Quality**: High overall quality, but **254 potential label errors** were detected using Confident Learning.
*   **Semantic Structure**: Classes are well-separated in the embedding space, but **Sci/Tech** exhibits the highest entropy (11.80), suggesting it is the most diverse/complex category.
*   **Robustness**: The dataset/models are **highly sensitive to typos** (5% character noise drops similarity to 0.77) but **robust to synonym replacements** (0.96).
*   **Entities**: Named entities are highly discriminative with low overlap between classes (Jaccard < 0.05).

## 2. Dataset Statistics
*   **Total Training Samples**: 102,000
*   **Class Balance**: Perfectly balanced (25% per class).
*   **Length Distribution**:
    *   Mean Word Count: **37.9 words**
    *   Max Word Count: 171 words
    *   The dataset consists of short, headline-centric news snippets.

## 3. Deep Semantic Analysis
### 3.1 Entropy & Diversity
Shannon Entropy measures the unpredictability of terms within a class. Higher entropy implies a richer or more chaotic vocabulary.

| Class | Shannon Entropy | Interpretation |
| :--- | :--- | :--- |
| **Sci/Tech** | **11.81** | Most diverse vocabulary; likely hardest to classify. |
| **Sports** | 11.49 | Moderate diversity. |
| **Business** | 11.46 | Moderate diversity. |
| **World** | 11.40 | Most structured/repetitive vocabulary. |

### 3.2 Entity Overlap (Jaccard Similarity)
Measures how many named entities (Person, Org, GPE) are shared between classes.

| | Sci/Tech | World | Sports | Business |
| :--- | :--- | :--- | :--- | :--- |
| **Sci/Tech** | 1.00 | 0.034 | 0.022 | 0.051 |
| **World** | 0.034 | 1.00 | 0.035 | 0.048 |
| **Sports** | 0.022 | 0.035 | 1.00 | 0.021 |
| **Business** | **0.051** | 0.048 | 0.021 | 1.00 |

*   **Observation**: Overlap is extremely low (< 5%), confirming that Named Entities are strong discriminators for this dataset.
*   **Highest Overlap**: Business & Sci/Tech (0.051), likely due to tech companies (Microsoft, Google) appearing in financial contexts.

## 4. Data Quality & Label Noise
Using **Cleanlab** with `all-MiniLM-L6-v2` embeddings, we identified **254 potential label issues**.

**Examples of Mislabeling:**
1.  *Text*: "Governments react differently to acts of terror..."
    *   **Given Label**: Business
    *   **Predicted**: World (Prob: 0.99)
    *   *Analysis*: Clear mislabeling; this is a political/world news snippet.
2.  *Text*: "Mladin Release From Road Atlanta..."
    *   **Given Label**: Sci/Tech
    *   **Predicted**: Sports (Prob: 0.98)
    *   *Analysis*: "Road Atlanta" and "Superbike Championship" are clearly Sports.
3.  *Text*: "IT that adapts to users' requirements..."
    *   **Given Label**: Business
    *   **Predicted**: Sci/Tech (Prob: 0.99)
    *   *Analysis*: Discusses IT and software; likely Sci/Tech, though Business context is possible.

**Recommendation**: These 254 samples should be either removed or relabeled before training the final model to prevent noise overfitting.

## 5. Robustness Profiling
We tested the semantic stability of the data against perturbations.

| Perturbation Type | Mean Cosine Similarity | Impact |
| :--- | :--- | :--- |
| **Word Synonym (10%)** | **0.9577** | Low Impact. Models should handle vocabulary shifts well. |
| **Char Typos (5%)** | **0.7689** | **High Impact**. The embedding space is brittle to character-level noise. |

**Implication**: Data augmentation pipelines MUST include character-level noise (e.g., `nlpaug` CharSwap) to improve robustness, as the base representation is fragile to typos.

## 6. Conclusion & Next Steps
The AG News dataset is clean and balanced but contains a small but high-confidence set of mislabeled examples. The primary challenge will be handling the short text length (mean 38 words) and ensuring robustness against noisy text (typos).

**Action Items:**
1.  **Preprocessing**: Implement character-level augmentation.
2.  **Cleaning**: Filter out the identified 254 mislabeled samples.
3.  **Modeling**: Focus on models that handle short text well (e.g., BERT, DistilBERT) and consider Entity-aware mechanisms given the high discriminative power of NER.
