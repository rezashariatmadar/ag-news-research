I\'ll review both frameworks and integrate the most valuable elements
into an enhanced, consolidated version. Let me create the definitive
framework that combines the best practices from all sources.

# **Ultimate News Classification Framework: AG News Dataset**

## **Complete 7-Step Implementation Guide**

## **Step 1: Project Setup, Data Acquisition & Evaluation Protocol**

### **1.1 Project Scope Definition**

**Objectives & Success Criteria**

-   **Primary metrics**: Macro-F1 (equal weight to all 4 classes) and
    > overall accuracy

-   **Secondary metrics**: Weighted-F1, per-class precision/recall/F1,
    > log-loss (calibration), confusion entropy, top-k accuracy

-   **Performance targets**:

    -   Linear models on TF-IDF: 0.84-0.90 macro-F1 (literature
        > baseline)

    -   Deep learning models: 0.88-0.93 macro-F1

    -   Ensemble: 0.90+ macro-F1

-   **Computational constraints**: Document inference latency (\<100ms
    > for classical, \<500ms for transformers), memory footprint,
    > training time

**Data Quality Strategy**

-   **AG News is cleaner** than 20 Newsgroups (no headers/footers/quotes).
-   Focus on **text normalization** (HTML tags, special characters) rather than metadata stripping.

## **1.2 Dataset Loading & Splits Strategy**

**Data Acquisition**

-   Use `datasets.load_dataset("sh0416/ag_news")` from Hugging Face.
-   **Train split**: 120,000 samples.
-   **Test split**: 7,600 samples.
-   **Classes**: 4 balanced categories: World (0), Sports (1), Business (2), Sci/Tech (3).

**Evaluation Protocol (Critical - Lock This Early)**

-   **Official test set**: Keep as untouched final holdout (use
    > subset=\'test\')

-   **Training set**: Apply 10-fold Stratified Cross-Validation for all
    > model selection

-   **Validation split**: Hold out 15-20% from training for
    > hyperparameter tuning and early stopping

-   **Nested CV**: For unbiased hyperparameter optimization, use inner
    > 5-fold CV within each of 10 outer folds (computationally expensive
    > but rigorous)

-   Collect **out-of-fold (OOF) predictions** from all CV folds for
    > meta-learning and error analysis

## **1.3 Reproducibility Infrastructure**

**Version Control & Tracking**

-   Fix random seeds globally: NumPy, scikit-learn, PyTorch, TensorFlow,
    > Python random

-   Document exact versions: scikit-learn, PyTorch, TensorFlow, CUDA,
    > cuDNN, transformers library

-   **Experiment tracking**: Set up MLflow, Weights & Biases, or
    > TensorBoard from day one

    -   Log: preprocessing config, features, hyperparameters, metrics
        > per fold, training curves, artifacts

-   **Code organization**: Modular structure (see Step 7 for details)

-   **Hardware specs**: Document CPU/GPU, RAM, training times for
    > reproducibility assessment

## **1.4 Initial Data Assessment**

**Basic Statistics**

-   Class distribution: Verify perfect balance (30,000 train / 1,900 test per class).

-   Train/test split ratio: 120,000 train / 7,600 test.

-   Memory footprint: Raw text \~50-100MB, vectorized features
    > \~500MB-2GB depending on vocabulary

-   Document length ranges: Characters, words, sentences per category

**Metadata Exploration**

-   Examine data.filenames patterns (file paths encode category
    > structure)

-   Map data.target indices to data.target_names

-   **World**: Global politics, conflicts, international relations.

    -   **Sports**: Game results, athlete news, team updates.

    -   **Business**: Markets, earnings, corporate news, economy.

    -   **Sci/Tech**: Technology launches, scientific discoveries, space, internet.

**Visualizations**

-   Bar chart: Class distribution (samples per category) with
    > color-coding by hierarchy

-   Train/test split verification: Side-by-side comparison

-   Schematic diagram: Evaluation protocol flowchart (CV folds,
    > validation, test holdout)

**Deliverables**

-   Written experimental plan document (metrics, splits, preprocessing
    > variants)

-   Experiment tracking template initialized

-   Data loading notebook with initial statistics

## **Step 2: Extensive Exploratory Data Analysis (EDA)**

## **2.1 Document-Level Statistics**

**Text Length Analysis**

-   **Distributions**: Character count, word count, sentence count per
    > document

    -   Compute: mean, median, std dev, min, max, quartiles, IQR

    -   Identify outliers: Documents \<50 words (noise?) or \>5000 words
        > (concatenated threads?)

-   **Per-category breakdown**: Violin plots, box plots, histograms by
    > newsgroup

    -   Hypothesis: Technical groups (comp.*, sci.*) may have longer,
        > more detailed posts

-   **Impact assessment**: Does document length correlate with
    > classification difficulty?

**Duplicate Detection**

-   **Exact duplicates**: Hash raw text (MD5/SHA256), count collisions
    > within and across splits

-   **Near-duplicates**: MinHash or Jaccard similarity on character
    > 5-grams; flag pairs \>0.9 similarity

-   **Cross-split contamination**: Check if test documents appear (even
    > partially) in training

-   **Heavy quoting**: Measure % of lines starting with \"\>\"; high
    > values indicate copy-paste

## **2.2 Vocabulary & Token Analysis**

**Global Vocabulary**

-   Total unique tokens across corpus (after basic tokenization)

-   Vocabulary growth curve: Unique words vs number of documents
    > (Heap\'s Law)

-   Hapax legomena: Words appearing only once (typically 40-50% of
    > vocabulary)

-   Zipf\'s law verification: Log-log plot of rank vs frequency

**Per-Category Vocabulary**

-   Vocabulary size per newsgroup

-   Vocabulary overlap heatmap: Jaccard similarity between category
    > vocabularies

    -   Expected: High overlap between comp.sys.mac vs comp.sys.ibm;
        > rec.autos vs rec.motorcycles

-   Category-specific vocabulary: Words unique to each category or
    > heavily concentrated in one

**N-gram Patterns**

-   **Top unigrams, bigrams, trigrams**: Globally and per category

-   **Discriminative n-grams**: Use TF-IDF, log-odds ratio, or pointwise
    > mutual information (PMI) to rank

    -   Example: \"forsale\" → misc.forsale; \"jesus\" →
        > talk.religion/soc.religion.christian; \"encrypt\" → sci.crypt

-   **Character n-grams (3-6)**: Capture stylistic patterns, typos,
    > technical notation (e.g., \"\_\_\_\", \"====\")

**2.3 Metadata & Leakage Assessment**

-   **Note**: AG News is largely free of the header/footer leakage found in 20 Newsgroups.
-   **Focus**: Check for HTML tags (`<br>`, `&quot;`) or concatenated title/description artifacts.

## **2.4 Category Relationship Analysis**

**Inter-Class Similarity**

-   **TF-IDF + Dimensionality Reduction**:

    -   Fit TfidfVectorizer on training data

    -   Apply TruncatedSVD (50-100 components) for LSA (Latent Semantic
        > Analysis)

    -   Visualize with UMAP (2D) and t-SNE (2D): Color by category, mark
        > confusion-prone pairs

-   **Centroid cosine similarity**: Compute average TF-IDF vector per
    > category; heatmap of pairwise similarities

    -   Expected confusions:

        -   **Business ↔ Sci/Tech**: Tech companies (Microsoft, Google) appear in both (earnings vs product launch).

        -   **World ↔ Business**: Economic policy vs political impact.

**Hierarchical Clustering**

-   Dendrogram of categories based on vocabulary or TF-IDF similarity

-   Identify natural category groupings vs official hierarchy

**Statistical Association Tests**

-   **Chi-square test**: Term-category contingency table; identify top
    > 100 discriminative terms per class

-   **Mutual information**: Information gain per term; compare to
    > chi-square rankings

-   **ANOVA F-statistic**: For continuous features (e.g., document
    > length, vocabulary richness)

## **2.5 Linguistic & Stylistic Features**

**Part-of-Speech (POS) Analysis**

-   Use spaCy or NLTK to tag: Noun, verb, adjective, adverb, etc.

-   POS distribution per category (bar charts): Hypothesis - scientific
    > groups use more nouns/adjectives; discussion groups more
    > verbs/adverbs

-   POS tag n-grams (e.g., \"NOUN NOUN\" vs \"VERB ADV\")

**Named Entity Recognition (NER)**

-   Extract: PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PRODUCT names

-   Top entities per category (word clouds or frequency tables)

-   Entity density: Entities per 100 words by category

**Readability & Complexity Metrics**

-   Flesch Reading Ease, Flesch-Kincaid Grade Level, Gunning Fog Index

-   Lexical diversity: Type-token ratio (TTR), moving average TTR
    > (MATTR)

-   Average word length, sentence length, syllable count

-   Hypothesis: Technical/scientific posts have lower readability,
    > higher complexity

**Sentiment & Subjectivity** (Optional)

-   Polarity scores using TextBlob or VADER

-   Distribution by category: Are some groups more opinionated
    > (talk.politics.*) vs neutral (sci.*)?

-   Subjectivity scores: Objective (factual) vs subjective (opinion)
    > content ratio

## **2.6 Quality & Anomaly Detection**

**Data Quality Issues**

-   Empty or near-empty documents (\<10 characters after whitespace
    > removal)

-   Non-English content: Language detection (langdetect library); flag
    > outliers

-   Encoding errors: Broken Unicode, mojibake patterns (e.g., \"Ã©\"
    > instead of \"é\")

-   Malformed structure: Missing text, only headers/signatures

**Outlier Documents**

-   Statistical outliers: Documents \>3 std dev from mean length

-   Topic outliers: Documents with very low similarity to category
    > centroid

-   Manual review: Inspect 10-20 outliers per category for mislabeling
    > or contamination

## **2.7 Comprehensive Visualization Suite**

**Distribution Plots**

-   Histograms: Word count, character count (log scale if needed)

-   Violin plots: Length distribution by category (shows density)

-   Box plots: Comparison across category groups (comp.\* vs rec.\* vs
    > sci.\*)

**Relationship Visualizations**

-   Heatmaps: Vocabulary overlap, centroid similarity, confusion
    > predictions (from dummy model)

-   Scatter plots: Document length vs vocabulary richness, colored by
    > category

-   2D embeddings: UMAP/t-SNE of TF-IDF vectors (interactive Plotly
    > version recommended)

**Word Frequency**

-   Word clouds: Top 50-100 words per category (size by TF-IDF weight)

-   Bar charts: Top 20 discriminative words per category with chi-square
    > scores

-   Treemaps: Hierarchical visualization of category → top words

**Interactive Dashboard** (Optional but Recommended)

-   Streamlit or Dash app: Select category → view statistics, word
    > clouds, sample documents

-   Document explorer: Search/filter by length, similarity, metadata
    > presence

## **2.8 Deep Semantic Manifold Analysis**
*Instead of TF-IDF (sparse), analyze how modern Dense Embeddings (BERT/RoBERTa) perceive the data **before** fine-tuning. This predicts transfer learning difficulty.*

**Embedding Space Geometry**
* **Isotropy Measurement**: Calculate the average cosine similarity of random pairs of sentence embeddings.
    * *Issue:* If highly anisotropic (all vectors point in a narrow cone), the model will struggle to distinguish nuances.
* **Intra-Class Variance**: Calculate the average Euclidean distance from the class centroid *within* the dense embedding space.
    * *Hypothesis:* "World" news usually has higher variance (looser semantic cluster) than "Sports" (tighter semantic cluster).
* **Out-of-Distribution (OOD) Simulation**: Train a One-Class SVM on the embeddings of 3 classes and test on the 4th. This measures how "distinct" a class is in the latent space.

## **2.9 Unsupervised Sub-Topic Discovery (Latent Variable Analysis)**
*AG News has 4 coarse classes, but each contains dozens of hidden sub-topics. You need to unearth them to understand class coherence.*

**Topic Modeling (BERTopic / LDA)**
* **Sub-cluster Identification**: Run BERTopic (or Hierarchical DBSCAN on embeddings) to find granular topics.
    * *Example:* Does the "Business" class split cleanly into *Earnings*, *Mergers*, and *Macro-Econ*? Or is it a mess?
* **Topic Coherence Scores (c_v)**: Calculate how semantically consistent the sub-topics are.
* **Class-Topic Entropy**:
    * Compute $H(T|C)$: The entropy of Topics given a Class.
    * *Goal:* Low entropy means a Class is composed of very few, distinct topics. High entropy means the class is a "catch-all" bucket (noisy).

## **2.10 Syntactic & Rhetorical Structure Analysis**
*News articles (especially headlines) follow specific syntactic rules ("Headlinese"). Analyze the structure, not just the words.*

**Dependency Parsing**
* **Tree Depth Analysis**: Use a dependency parser (spaCy/Stanza) to calculate the maximum depth of the syntactic tree.
    * *Hypothesis:* "Sci/Tech" might have deeper nesting (complex clauses) than "Sports".
* **Root Verb Distribution**: Extract the root of the dependency tree for every document.
    * *Action:* Compare root verbs across classes. (e.g., Sports uses "defeat", "win"; Business uses "plunge", "soar").
* **Voice Detection**: Ratio of Passive vs. Active voice constructions.

## **2.11 Data-Centric AI: Label Noise & Confident Learning**
*Assume the ground truth labels are imperfect. This is the "Systematic" part of modern ML.*

**Confident Learning (Cleanlab Approach)**
* **Cross-Validation Probability Check**: Train a simple logistic regression on fixed embeddings via 5-fold CV. Get out-of-sample probability predictions.
* **Off-Diagonal Analysis**: Identify samples where $P(True\_Label) < Threshold$ and $P(Other\_Label) > Threshold$.
    * *Deliverable:* A list of "Potentially Mislabelled Samples" to manually inspect.
* **Label Quality Score**: Assign a trust score to every sample. Plot the distribution of trust scores per category.

## **2.12 Information Theoretic Metrics**
*Measure the "information density" of the text.*

**Perplexity & Entropy**
* **Zero-Shot Perplexity**: Run the dataset through a small causal LLM (e.g., GPT-2 or TinyLlama).
    * *Goal:* High perplexity indicates anomalous text, garbled data, or highly creative/irregular language.
* **Shannon Entropy of Terms**:
    * Calculate the entropy of term distributions per class.
    * *Insight:* Classes with higher entropy are harder to learn because the vocabulary is less predictive.

## **2.13 Augmentation & Robustness Profiling**
*Determine "how much" and "what kind" of data augmentation is needed.*

**Perturbation Sensitivity**
* **Typo Simulation**: Randomly swap 5% of characters (simulating keyboard noise). Measure how much the embedding vector shifts (Cosine distance between Original and Perturbed).
* **Synonym Replacement Sensitivity**: Replace nouns/verbs with WordNet synonyms.
    * *Goal:* If the embedding shifts drastically with simple synonyms, the data representation is brittle.

## **2.14 Correlation with External Knowledge (Entity Linking)**
*AG News relies heavily on entities (Microsoft, Yankees, Iraq). Check the knowledge graph density.*

**Entity Linking & Wikification**
* **Knowledge Base Coverage**: Map entities to Wikidata IDs.
* **Entity Overlap Matrix**:
    * Do "Business" and "Sci/Tech" share the same entities (e.g., "Google")?
    * *Metric:* Jaccard similarity of the *Entity Set* (not word set) between classes.
    * *Why:* If entity overlap is high but class labels differ, the model must learn *context*, not just keywords.

---

### **Updated Deliverable Structure**
Your final report should now include a section called **"3.0 Latent & Structural Diagnostics,"** containing:
1.  **The "Confusion Potential" Heatmap:** Based on embedding isotropy and overlap.
2.  **The "Dirty Data" List:** The top 100 mislabeled candidates found via Confident Learning.
3.  **Sub-Topic Hierarchy Map:** Visualizing the breakdown of the 4 classes into 50+ semantic clusters.

**Deliverables**

-   Comprehensive EDA report (15-25 pages with visualizations)

-   Key findings document:

    -   Confusion-prone category pairs

    -   High-signal features (discriminative terms, structural patterns)

    -   Leakage assessment: Estimated impact of headers/quotes (e.g.,
        > \"Removing headers reduces accuracy by 5-8%\")

    -   Recommended preprocessing pipeline

-   Annotated visualizations saved in results/figures/eda/

## **Step 3: Data Preprocessing & Text Engineering**

## **3.1 Text Cleaning Pipeline Design**

**Modular Cleaning Functions** (Test Each Component via CV)

**Basic Cleaning**

-   **Lowercase conversion**: Optional; preserve case for character
    > n-grams to capture acronyms, shouting (ALL CAPS)

-   **Whitespace normalization**: Replace tabs, multiple spaces,
    > newlines with single space

-   **HTML/URL handling**:

    -   Strategy 1: Remove completely

    -   Strategy 2: Replace with tokens \<URL\>, \<EMAIL\> (preserves
        > structural signal)

    -   Strategy 3: Keep as-is (character n-grams may capture domain
        > patterns)

-   **Special characters**:

    -   Punctuation: Keep for char n-grams; remove for word-only models

    -   Numbers: Replace with \<NUM\> token or keep; test both

-   **Encoding fixes**: Handle Unicode errors, decode HTML entities (& →
    > &)

**Header/Footer/Quote Removal**

-   *Not applicable for AG News.* Skip this step.

**Contraction & Abbreviation Expansion**

-   \"don\'t\" → \"do not\", \"can\'t\" → \"cannot\" using contractions
    > library

-   Common abbreviations: \"msg\" → \"message\", \"pls\" → \"please\"

-   Evaluate impact: May help word-based models, hurt char n-grams

## **3.2 Tokenization Strategies**

**Word-Level Tokenization** (Compare Multiple Approaches)

1.  **Whitespace tokenization**: text.split() - fastest, simplest
    > baseline

2.  **NLTK word_tokenize**: Handles punctuation better (e.g., \"don\'t\"
    > → \"do\", \"n\'t\")

3.  **spaCy tokenization**: Linguistic rules, handles contractions well

4.  **Custom regex**: r\'\\b\\w+\\b\' or r\'\\w+\' depending on needs

5.  **Subword tokenization** (for deep learning):

    -   Byte-Pair Encoding (BPE): Hugging Face tokenizers, better OOV
        > handling

    -   WordPiece: BERT tokenizer, handles rare words

    -   SentencePiece: Language-agnostic, handles multilingual if needed

**Sentence Tokenization** (For Context-Aware Models)

-   NLTK sent_tokenize or spaCy sentence segmenter

-   Preserve sentence boundaries for hierarchical models (sentence →
    > document encoding)

**Token Statistics After Tokenization**

-   Average tokens per document by method

-   Vocabulary size comparison

-   OOV rate for pretrained embeddings

## **3.3 Text Normalization Techniques**

**Stop Word Removal**

-   **Standard lists**: NLTK English stop words (179 words), spaCy (326
    > words)

-   **Domain-specific stop words**: Identify from corpus (high
    > frequency, low discrimination)

    -   Example: \"article\", \"post\", \"writes\", \"lines\" (newsgroup
        > jargon)

-   **Adaptive stop words**: Remove top 0.1% most frequent words not in
    > standard lists

-   **Evaluation**: Test impact on macro-F1; removing stop words can
    > hurt performance on 20NG

    -   Recommendation: Keep stop words initially; test removal as
        > ablation

**Stemming vs Lemmatization**

-   **Porter Stemmer**: Aggressive, fast, but can conflate unrelated
    > words (\"university\" → \"univers\")

-   **Snowball Stemmer**: Improved Porter, language-specific rules

-   **Lancaster Stemmer**: Most aggressive, often too aggressive

-   **WordNet Lemmatizer**: Context-aware, preserves meaning (needs POS
    > tags)

-   **spaCy Lemmatizer**: Fast, integrates with pipeline

-   **Comparison strategy**: Run 10-fold CV with no normalization,
    > stemming, lemmatization; quantify trade-offs

    -   Expected: Lemmatization slightly better than stemming; both may
        > hurt char n-gram models

## **3.4 Feature Extraction & Vectorization**

**Traditional Sparse Representations**

**1. Bag-of-Words (CountVectorizer)**

-   **Parameters to tune**:

    -   ngram_range: (1,1), (1,2), (1,3) - unigrams, bigrams, trigrams

    -   max_features: 5000, 10000, 20000, 50000, None

    -   min_df: Minimum document frequency - 2, 3, 5, 10 (absolute) or
        > 0.001, 0.01 (relative)

    -   max_df: Maximum document frequency - 0.9, 0.95, 0.99 (filter
        > common words)

    -   binary: True (binary presence) vs False (counts)

-   **Vocabulary analysis**: Plot vocabulary size vs min_df/max_df

-   **Use case**: Baseline for Naive Bayes

**2. TF-IDF (TfidfVectorizer) - Primary Feature Representation**

-   **Core parameters**:

    -   ngram_range: (1,2) or (1,3) typically best for 20NG

    -   max_features: 20000-100000 (balance sparsity vs information)

    -   min_df: 3-5 (filter noise from rare words)

    -   max_df: 0.90-0.95 (filter corpus-wide common words)

    -   sublinear_tf: True (apply log scaling: 1 + log(tf)) -
        > **recommended\
        > **

    -   norm: \'l2\' (default, L2 normalization) vs \'l1\' vs None

    -   use_idf: True (default) vs False (just TF)

-   **Character-level TF-IDF**:

    -   analyzer=\'char\', ngram_range=(3,5) or (3,6)

    -   Captures stylistic, orthographic patterns; robust to typos

    -   Especially strong for 20NG due to formatting/style differences

-   **Comparison with pre-vectorized**:

    -   fetch_20newsgroups_vectorized() uses specific TF-IDF settings

    -   Verify: Load pre-vectorized, extract features manually, compare
        > results

**3. Hashing Vectorizer (Memory-Efficient Alternative)**

-   HashingVectorizer: Fixed-size feature space, no vocabulary stored

-   **Parameters**: n_features=2\*\*18 (262,144), ngram_range, norm

-   **Trade-off**: Saves memory, enables online learning, but loses
    > interpretability (no inverse mapping)

**Hybrid Representations (Often Strongest for Linear Models)**

-   **Word + Character TF-IDF concatenation**:

    -   Fit separate TfidfVectorizer for word (1,2)-grams and char
        > (3,5)-grams

    -   Horizontally stack (scipy.sparse.hstack) sparse matrices

    -   Typical dimensions: 50k word features + 50k char features = 100k
        > total

-   **Weighting strategies**: Test equal weight vs scaling (e.g., 0.7 \*
    > word + 0.3 \* char)

**Feature Selection (Post-Vectorization)**

-   **Chi-square (chi2)**: Select top k features most associated with
    > target

    -   Use SelectKBest(chi2, k=20000) on TF-IDF features

    -   Fast, effective; works only with non-negative features

-   **Mutual Information (MI)**: Information gain per feature

    -   mutual_info_classif: Captures non-linear relationships

    -   Slower than chi2 but sometimes better

-   **L1 Regularization (Lasso)**: Embedded feature selection

    -   LogisticRegression(penalty=\'l1\', solver=\'saga\') or
        > LinearSVC(penalty=\'l1\', dual=False)

    -   Automatically zeros coefficients of unimportant features

-   **Recursive Feature Elimination (RFE)**:

    -   RFE(estimator, n_features_to_select): Backward elimination

    -   Computationally expensive; use only for final tuning

-   **Variance Threshold**: Remove features with variance \< threshold
    > (rarely useful for text)

**Dimensionality Reduction for Dense Features**

-   **Truncated SVD (LSA)**:

    -   TruncatedSVD(n_components=100-500) on TF-IDF matrix

    -   Creates dense low-dimensional semantic space

    -   Speeds up training; sometimes improves generalization

    -   Plot explained variance ratio to choose components

-   **PCA**: Requires dense input; use on small datasets or after SVD

-   **NMF (Non-negative Matrix Factorization)**:

    -   Topic modeling approach; produces interpretable components

    -   NMF(n_components=50-100) on TF-IDF

    -   Useful for visualization and feature understanding

-   **LDA (Linear Discriminant Analysis)**:

    -   Supervised dimensionality reduction; max components =
        > n_classes - 1 = 19

    -   Can improve linear classifier decision boundaries

## **3.5 Auxiliary Feature Engineering (Optional Enhancements)**

**Document-Level Statistical Features**

-   Length features: Character count, word count, sentence count

-   Complexity: Average word length, average sentence length, vocabulary
    > richness (TTR)

-   Structural: Punctuation density (punctuation / total chars),
    > uppercase ratio, digit ratio

-   Metadata presence: Binary flags for URL/email/phone presence

**Linguistic Features**

-   POS distribution: Noun ratio, verb ratio, adjective ratio (from
    > spaCy tagging)

-   Dependency depth: Maximum parse tree depth (complexity indicator)

-   Syntactic patterns: Count of specific POS sequences (e.g., NOUN
    > NOUN, ADJ NOUN)

**Domain-Specific Features (Newsgroup-Specific)**

-   Quote indicators: Lines starting with \"\>\", quote depth (\> vs
    > \>\> vs \>\>\>)

-   Signature markers: Presence of \"\--\", \"wrote:\", \"writes:\"

-   Timestamp patterns: Date/time mentions

-   Thread depth: \"Re:\" count in subject

**Concatenation Strategy**

-   Add as dense columns to sparse TF-IDF matrix using
    > scipy.sparse.hstack

-   Scale auxiliary features (StandardScaler) before concatenation to
    > balance magnitude

## **3.6 Embedding-Based Features (Dense Representations)**

**Static Word Embeddings**

-   **Word2Vec**:

    -   **Pretrained**: Google News 300d vectors (\~3.6M vocabulary)

    -   **Custom-trained**: gensim.models.Word2Vec on 20NG corpus

        -   Parameters: vector_size=100-300, window=5, min_count=5, sg=0
            > (CBOW) or sg=1 (Skip-gram)

    -   **Document representation**: Average word vectors (simple),
        > TF-IDF weighted average (better)

-   **GloVe**:

    -   **Pretrained**: 6B tokens, 50d/100d/200d/300d (from Common
        > Crawl)

    -   Download, load with gensim or manual loading

    -   Same averaging strategies as Word2Vec

-   **FastText**:

    -   **Advantage**: Handles OOV words via character n-grams

    -   **Pretrained**: Available from Facebook Research

    -   **Custom-trained**: gensim.models.FastText on corpus

    -   Better for misspellings, rare words common in newsgroups

**Document Embedding Strategies**

-   **Simple average**: Mean of word vectors in document (ignores word
    > order)

-   **Weighted average**: TF-IDF weights \* word vectors (emphasizes
    > important words)

-   **Max/Min pooling**: Element-wise max or min across word vectors
    > (captures extremes)

-   **Concatenation**: \[mean, max, min\] to capture multiple statistics

-   **Doc2Vec / Paragraph Vectors**:

    -   gensim.models.Doc2Vec: Learn document-level embeddings

    -   Requires training on corpus; slower but captures document
        > context

**Visualization**

-   t-SNE/UMAP of word embeddings: Plot top 500 words colored by
    > associated category

-   Document embedding space: 2D projection colored by category; assess
    > separability

## **3.7 Preprocessing Pipeline Variants to Compare**

**Pipeline Configurations** (Test All via 10-Fold CV)

1.  **Baseline**: TF-IDF word (1,2)-grams, no stop word removal, no
    > stemming

2.  **Char-enhanced**: Word (1,2) + Char (3,5) TF-IDF concatenation

3.  **Stemmed**: Word (1,2) TF-IDF with Porter stemming

4.  **Lemmatized**: Word (1,2) TF-IDF with lemmatization

5.  **Feature-selected**: Chi2 top 20k from word (1,3) TF-IDF

6.  **Dimensionality-reduced**: Word (1,2) TF-IDF → SVD 300 components

7.  **Embedding-based**: TF-IDF weighted GloVe 300d average

8.  **Hybrid**: Word+Char TF-IDF + auxiliary features + embedding
    > average

**Deliverables**

-   Preprocessing pipeline code (modular, reusable functions)

-   Comparison table: Macro-F1 for each configuration on validation set

-   Feature space statistics: Dimensions, sparsity, memory usage

-   Saved vectorizers/transformers (joblib) for reproducibility

-   Visualization: Validation curves (performance vs min_df, vs
    > max_features, vs n_components)

## **Step 4: Baseline Modeling with 10-Fold Cross-Validation**

## **4.1 Cross-Validation Framework Setup**

**Stratified K-Fold Design**

-   **Configuration**: StratifiedKFold(n_splits=10, shuffle=True,
    > random_state=42)

-   **Purpose**: Ensure class balance in every fold; 10 folds provides
    > robust variance estimates

-   **Separate validation holdout**: Extract 15-20% from training set
    > before CV for hyperparameter selection and early stopping (deep
    > learning)

-   **OOF Predictions**: Collect out-of-fold predictions for:

    -   Error analysis and confusion matrix on full training set

    -   Stacking meta-learner training

    -   Ensemble weighting optimization

**Nested CV for Unbiased Hyperparameter Selection** (Optional,
Compute-Intensive)

-   **Outer loop**: 10-fold CV for performance estimation

-   **Inner loop**: 5-fold CV within each outer fold for hyperparameter
    > tuning

-   **Total fits**: 10 \* (grid_size \* 5) models - use only for final
    > model selection

-   **Alternative**: Single train/validation split for tuning, then
    > 10-fold CV with best params

## **4.2 Dummy Baselines (Establish Floor)**

**DummyClassifier Strategies**

-   **Most frequent**: Always predict most common class - expected \~5%
    > accuracy (1/20)

-   **Stratified**: Sample predictions proportional to class
    > distribution - \~5% accuracy

-   **Uniform**: Random prediction with equal probability per class -
    > \~5% accuracy

-   **Purpose**: Sanity check; any real model must exceed this
    > significantly

## **4.3 Classical Machine Learning Models**

**1. Naive Bayes Family** (Fast, Strong Baselines)

-   **Multinomial Naive Bayes** (MultinomialNB):

    -   **Best for**: Count or TF-IDF features (non-negative)

    -   **Hyperparameter**: alpha (additive smoothing) - test \[0.01,
        > 0.1, 0.5, 1.0, 2.0\]

    -   **Expected performance**: 0.70-0.82 macro-F1 depending on
        > features

    -   **Advantage**: Extremely fast training/inference; probabilistic
        > outputs

-   **Complement Naive Bayes** (ComplementNB):

    -   **Designed for**: Imbalanced datasets; models complement of each
        > class

    -   **Often better** than MultinomialNB on text; same
        > hyperparameters

-   **Bernoulli Naive Bayes** (BernoulliNB):

    -   **Best for**: Binary (presence/absence) features

    -   Use with CountVectorizer(binary=True)

-   **Gaussian Naive Bayes**: Not recommended for text (requires dense,
    > normalized features)

**2. Linear Models** (Top Performers on 20NG)

-   **Logistic Regression** (LogisticRegression):

    -   **Solver options**:

        -   \'liblinear\': Good for small datasets, supports L1/L2

        -   \'saga\': Supports L1/L2/ElasticNet, scales to large data

        -   \'lbfgs\': Fast for L2, multiclass optimized

    -   **Regularization**:

        -   penalty=\'l2\': Ridge, default; tune C=\[0.01, 0.1, 1.0,
            > 10.0, 100.0\] (inverse regularization)

        -   penalty=\'l1\': Lasso, feature selection; requires
            > solver=\'saga\' or \'liblinear\'

        -   penalty=\'elasticnet\': L1 + L2 mix; requires
            > solver=\'saga\' and l1_ratio

    -   **Multi-class**: multi_class=\'ovr\' (one-vs-rest) or
        > \'multinomial\' (softmax)

    -   **Class weighting**: class_weight=\'balanced\' if any imbalance

    -   **Convergence**: max_iter=1000 or higher to ensure convergence

    -   **Expected performance**: 0.82-0.88 macro-F1 (top classical
        > performer)

    -   **Advantages**: Probabilistic, calibrated outputs; interpretable
        > coefficients

-   **Linear SVM** (LinearSVC):

    -   **Loss**: loss=\'squared_hinge\' (default, L2-loss) or \'hinge\'
        > (L1-loss)

    -   **Regularization**: C=\[0.001, 0.01, 0.1, 1.0, 10.0\] (inverse
        > regularization, like Logistic)

    -   **Penalty**: penalty=\'l2\' (default) or \'l1\' (feature
        > selection, requires dual=False)

    -   **Multi-class**: One-vs-rest by default

    -   **Expected performance**: 0.84-0.90 macro-F1 (**often best
        > classical model on 20NG**)

    -   **Advantage**: Strong maximum-margin decision boundaries;
        > handles high-dimensional sparse data well

    -   **Note**: Does not output calibrated probabilities; use
        > CalibratedClassifierCV wrapper if needed

-   **SGDClassifier** (SGDClassifier):

    -   **Loss functions**:

        -   \'hinge\': Linear SVM

        -   \'log_loss\': Logistic regression

        -   \'modified_huber\': Smooth hinge, robust to outliers,
            > outputs probabilities

    -   **Penalty**: \'l2\', \'l1\', \'elasticnet\'

    -   **Alpha**: Regularization strength - \[1e-5, 1e-4, 1e-3, 1e-2\]

    -   **Learning rate**: learning_rate=\'optimal\' (default),
        > \'invscaling\', \'constant\'

    -   **Early stopping**: early_stopping=True,
        > validation_fraction=0.1, n_iter_no_change=5

    -   **Advantage**: Very efficient for large datasets; online
        > learning capable

    -   **Expected performance**: Similar to LinearSVC/LogReg if tuned
        > properly

**3. Support Vector Machines with Kernels** (Optional, Expensive)

-   **SVC** (SVC):

    -   **Kernels**: \'linear\' (use LinearSVC instead for speed),
        > \'rbf\', \'poly\'

    -   **Best on subset**: Full 20NG too large; test on 2-5 categories

    -   **Hyperparameters**: C, gamma=\'scale\' or \'auto\' for RBF/poly

    -   **Expected**: Marginal gain over linear, not worth compute cost
        > for 20NG

**4. Nearest Neighbors** (Baseline, Usually Weak)

-   **K-Nearest Neighbors** (KNeighborsClassifier):

    -   **Parameters**: n_neighbors=\[3\]\[5\]\[7\]\[11\],
        > metric=\'cosine\' (good for text), weights=\'distance\'

    -   **Expected performance**: 0.60-0.75 macro-F1 (weak due to curse
        > of dimensionality)

    -   **Advantage**: Non-parametric, interpretable (see similar
        > training documents)

    -   **Disadvantage**: Slow inference on large training sets

**5. Tree-Based Models**

-   **Decision Tree** (DecisionTreeClassifier):

    -   **Purpose**: Weak baseline; compare to ensemble performance

    -   **Parameters**: max_depth=\[10, 20, None\],
        > min_samples_split=\[5\]\[10\]

    -   **Expected performance**: 0.50-0.65 macro-F1 (underfits
        > high-dimensional sparse data)

-   **Random Forest** (RandomForestClassifier):

    -   **Parameters**:

        -   n_estimators=\[100\]\[200\]\[300\]

        -   max_depth=\[None, 20, 30, 50\]

        -   min_samples_split=\[2\]\[5\]\[10\]

        -   max_features=\[\'sqrt\', \'log2\'\] (avoid overfitting)

    -   **Expected performance**: 0.70-0.80 macro-F1 (decent but usually
        > below linear models on text)

    -   **Advantage**: Feature importance scores; handles non-linear
        > interactions

    -   **Disadvantage**: Memory-intensive for sparse features; slower
        > than linear models

-   **Gradient Boosting Machines**:

    -   **XGBoost** (XGBClassifier):

        -   n_estimators=\[100\]\[200\], learning_rate=\[0.05, 0.1\],
            > max_depth=\[3\]\[5\]\[7\]

        -   subsample=\[0.8, 1.0\], colsample_bytree=\[0.8, 1.0\]

        -   Early stopping: early_stopping_rounds=10, eval_set

    -   **LightGBM** (LGBMClassifier):

        -   Similar params; often faster than XGBoost

        -   boosting_type=\'gbdt\' (gradient boosting) or \'dart\'
            > (dropout trees)

    -   **CatBoost** (CatBoostClassifier):

        -   Handles categorical features natively (less relevant for
            > text)

        -   iterations=\[100\]\[200\], learning_rate=\[0.05, 0.1\],
            > depth=\[4\]\[6\]

    -   **Expected performance**: 0.75-0.85 macro-F1 (competitive but
        > rarely beat tuned linear models)

    -   **Use case**: Ensemble diversity; strong for non-text auxiliary
        > features

## **4.4 Evaluation Metrics & Diagnostics**

**Core Metrics (Compute Per Fold, Aggregate Across Folds)**

-   **Accuracy**: Simple correctness; biased if classes unbalanced

-   **Macro-F1**: Average F1 across classes (equal weight) - **primary
    > metric for 20NG\
    > **

-   **Weighted-F1**: Average F1 weighted by class support

-   **Micro-F1**: Aggregates TP/FP/FN globally (equals accuracy for
    > multiclass)

-   **Per-class Precision, Recall, F1**: Identify weak/strong classes

-   **Cohen\'s Kappa**: Agreement beyond chance; useful for imbalanced
    > sets

-   **Matthews Correlation Coefficient (MCC)**: Balanced measure even
    > with imbalance

**Probabilistic Metrics** (For Models with Probability Outputs)

-   **Log-loss (Cross-entropy)**: Penalizes confident wrong predictions

-   **Brier score**: Mean squared error of predicted probabilities

-   **ROC-AUC**: One-vs-rest for each class; macro/micro/weighted
    > average

    -   Requires predict_proba or decision function

-   **Precision-Recall AUC**: Better for imbalanced classes than ROC

**Calibration Assessment**

-   **Reliability diagram**: Plot mean predicted probability vs
    > empirical accuracy in bins

-   **ECE (Expected Calibration Error)**: Weighted average of bin-wise
    > calibration errors

-   **Models to calibrate**: SVM (not probabilistic by default), some
    > boosted models

-   **Calibration methods**: CalibratedClassifierCV with
    > method=\'sigmoid\' (Platt) or \'isotonic\'

**Confusion Matrix Analysis**

-   Aggregate OOF predictions across all folds → single confusion matrix
    > on full training set

-   Heatmap visualization (seaborn): Annotate with counts or normalized
    > by true class

-   **Key insights**:

    -   Diagonal = correct predictions

    -   Off-diagonal = confusions; identify systematic errors

    -   Symmetry: Is class A → B confusion reciprocated by B → A?

**Learning Curves**

-   **Training size vs performance**: Plot macro-F1 vs fraction of
    > training data \[0.1, 0.3, 0.5, 0.7, 0.9, 1.0\]

-   **Purpose**: Diagnose underfitting (both train/val low, converge) vs
    > overfitting (train high, val plateaus low)

-   **Implementation**: learning_curve() from sklearn

**Validation Curves**

-   **Hyperparameter vs performance**: Fix other params; vary one (e.g.,
    > C, alpha, n_estimators)

-   **Plot**: X-axis = hyperparameter value (log scale if wide range),
    > Y-axis = mean macro-F1 ± std across folds

-   **Purpose**: Visualize optimal hyperparameter range and sensitivity

## **4.5 Statistical Significance Testing**

**Pairwise Model Comparison**

-   **Paired t-test**: Compare fold-wise macro-F1 scores between two
    > models

    -   Null hypothesis: No difference in means

    -   Use if fold scores are approximately normal

    -   scipy.stats.ttest_rel(model_a_scores, model_b_scores)

-   **Wilcoxon signed-rank test**: Non-parametric alternative to t-test

    -   Better for small sample sizes (10 folds) or non-normal
        > distributions

    -   scipy.stats.wilcoxon(model_a_scores, model_b_scores)

-   **McNemar\'s test**: Compare models on paired binary outcomes
    > (correct/incorrect per instance)

    -   Builds 2x2 contingency table from OOF predictions

    -   statsmodels.stats.contingency_tables.mcnemar

**Multiple Model Comparison**

-   **Friedman test**: Non-parametric ANOVA for multiple models across
    > folds

    -   Null: All models equivalent

    -   If rejected, proceed to post-hoc tests

    -   scipy.stats.friedmanchisquare(model1_scores, model2_scores,
        > \...)

-   **Nemenyi post-hoc test**: Pairwise comparisons after Friedman

    -   Controls family-wise error rate

    -   Implementation: scikit-posthocs.posthoc_nemenyi_friedman

**Effect Size**

-   **Cohen\'s d**: Standardized difference between two models

    -   Small (0.2), medium (0.5), large (0.8) thresholds

    -   Assess practical significance beyond p-values

## **4.6 Baseline Results Summary**

**Comparison Table** (Latex/Markdown Format)

  -----------------------------------------------------------------------------
  **Model**        **Mean       **Std   **Mean       **Training   **Inference
                   Macro-F1**   Dev**   Accuracy**   Time (s)**   Time (ms)**
  ---------------- ------------ ------- ------------ ------------ -------------
  Dummy            0.050        0.005   0.050        \<0.01       \<0.01
  (Stratified)                                                    

  MultinomialNB    0.XXX        0.XXX   0.XXX        X.X          X.X

  ComplementNB     0.XXX        0.XXX   0.XXX        X.X          X.X

  Logistic         0.XXX        0.XXX   0.XXX        XX.X         X.X
  Regression                                                      

  LinearSVC        0.XXX        0.XXX   0.XXX        XX.X         X.X

  SGDClassifier    0.XXX        0.XXX   0.XXX        XX.X         X.X

  Random Forest    0.XXX        0.XXX   0.XXX        XXX.X        XX.X

  XGBoost          0.XXX        0.XXX   0.XXX        XXX.X        XX.X

  \...             \...         \...    \...         \...         \...
  -----------------------------------------------------------------------------

**Visualizations**

-   **Box plots**: One per model showing distribution of 10 fold
    > macro-F1 scores

-   **Bar chart with error bars**: Mean ± std macro-F1 for all models;
    > sort by performance

-   **Confusion matrices**: Top 3 models (heatmaps)

-   **ROC curves**: Multiclass one-vs-rest for top 2-3 models (20 curves
    > per model, color by class)

-   **Precision-Recall curves**: Similar to ROC but better for
    > imbalanced classes

**Deliverables**

-   Cross-validation results CSV: Model, fold, metrics per fold

-   OOF predictions and true labels (for meta-learning)

-   Trained models saved (joblib): Best fold model or retrained on full
    > training set with best hyperparams

-   Baseline report document: Key findings, top models, confusion
    > analysis

-   Code: Reusable CV framework and evaluation utilities

## **Step 5: Feature Engineering Refinement, Hyperparameter Optimization & Ensembling**

## **5.1 Advanced Feature Engineering Iterations**

**Guided by Baseline Results**

-   **Error analysis findings**: Which categories are confused? What
    > features might help?

    -   Example: comp.sys.mac vs comp.sys.ibm confused → Add char
        > n-grams to capture brand names

    -   Example: rec.sport.baseball vs rec.sport.hockey → More
        > sport-specific vocabulary (team names, terminology)

**Feature Blending Strategies**

-   **Word + Char TF-IDF Weight Tuning**:

    -   Test: 0.5 \* word + 0.5 \* char, 0.6 \* word + 0.4 \* char, 0.7
        > \* word + 0.3 \* char

    -   Grid search or optimize on validation set

-   **Embedding Augmentation**:

    -   Concatenate: \[TF-IDF sparse (100k dim), GloVe average (300
        > dim)\]

    -   Test weighting: Scale embedding features before concatenation

-   **Auxiliary Feature Integration**:

    -   Add: Document length, POS ratios, quote density (5-20 features)

    -   Combine with sparse text features via scipy.sparse.hstack

**Domain-Specific Feature Engineering**

-   **Category-specific keyword dictionaries**:

    -   Example: sci.crypt → \"encrypt\", \"cipher\", \"key\",
        > \"crypto\"

    -   Create binary or count features for keyword presence

-   **Quote and signature features**:

    -   Ratio of quoted lines, signature presence (binary), quote depth

    -   Hypothesis: Discussion categories (talk.\*) have more quotes

-   **Technical notation features**:

    -   Regex patterns: Code snippets (indentation, keywords), equations
        > (LaTeX-like), ASCII art

**Feature Selection Refinement**

-   **Recursive Feature Elimination (RFE)**:

    -   Use top baseline model (e.g., LinearSVC) as estimator

    -   Eliminate bottom 10% features per iteration; track performance

    -   Stop when performance plateaus or starts declining

-   **Permutation Importance**:

    -   Shuffle each feature, measure drop in macro-F1

    -   Rank features by importance; remove bottom 20-30%

-   **Stability Selection**:

    -   Bootstrap sampling + feature selection (e.g., L1 Lasso); track
        > selection frequency

    -   Keep features selected in \>50% of bootstrap runs

## **5.2 Systematic Hyperparameter Optimization**

**Search Strategies**

**1. Grid Search** (GridSearchCV)

-   **Use when**: Small hyperparameter space (\<100 combinations)

-   **Example**: Logistic Regression\
    > \
    > python

param_grid = {

\'C\': \[0.1, 1.0, 10.0\],

\'penalty\': \[\'l1\', \'l2\'\],

\'solver\': \[\'saga\'\], *\# supports both L1/L2*

}

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5,
scoring=\'f1_macro\', n_jobs=-1)

-   

-   **Advantage**: Exhaustive; guaranteed to find best in grid

-   **Disadvantage**: Exponential growth in compute time

**2. Randomized Search** (RandomizedSearchCV)

-   **Use when**: Large hyperparameter space (\>100 combinations)

-   **Example**: Random Forest\
    > \
    > python

param_dist = {

\'n_estimators\': randint(100, 500),

\'max_depth\': \[None\] + list(range(10, 100, 10)),

\'min_samples_split\': randint(2, 20),

\'max_features\': \[\'sqrt\', \'log2\', 0.5\],

}

random = RandomizedSearchCV(RandomForestClassifier(), param_dist,
n_iter=100, cv=5, scoring=\'f1_macro\', n_jobs=-1)

-   

-   **Advantage**: Efficient exploration; can sample from continuous
    > distributions

-   **Recommendation**: Use for initial broad search, then Grid Search
    > around best region

**3. Bayesian Optimization** (Optuna, Hyperopt, Scikit-Optimize)

-   **Use when**: Expensive model training (e.g., deep learning, large
    > gradient boosting)

-   **Example with Optuna**:\
    > \
    > python

def objective(trial):

C = trial.suggest_loguniform(\'C\', 1e-3, 1e2)

penalty = trial.suggest_categorical(\'penalty\', \[\'l1\', \'l2\'\])

model = LogisticRegression(C=C, penalty=penalty, solver=\'saga\')

*\# CV evaluation\...*

return macro_f1

study = optuna.create_study(direction=\'maximize\')

study.optimize(objective, n_trials=100)

-   

-   **Advantage**: Intelligent exploration; focuses on promising
    > regions; fewer evaluations needed

-   **Best for**: 20-100+ hyperparameters, expensive objectives

**4. Halving Search** (HalvingGridSearchCV, HalvingRandomSearchCV)

-   **Use when**: Many candidates, limited compute

-   **Strategy**: Start with all candidates on small data subset;
    > progressively eliminate poor performers and increase data

-   **Advantage**: Fast early elimination of bad configurations

**Nested CV for Unbiased Estimates**

-   **Outer loop**: 10-fold Stratified CV (performance estimation)

-   **Inner loop**: 5-fold CV within each outer fold (hyperparameter
    > selection)

-   **Process**:

    1.  For each of 10 outer folds:

        -   Hold out fold as test

        -   Use remaining 9 folds: 5-fold inner CV with hyperparameter
            > search

        -   Select best hyperparameters from inner CV

        -   Train on all 9 outer training folds with best params

        -   Evaluate on outer test fold

    2.  Report: Mean ± std of 10 outer fold scores

-   **Advantage**: Unbiased performance estimate; hyperparameter
    > selection not contaminated by test set

-   **Disadvantage**: Computationally expensive (10 \* n_combinations \*
    > 5 fits)

-   **Recommendation**: Use for final 2-3 top models only

## **5.3 Probability Calibration**

**Why Calibrate?**

-   **Problem**: Some models (SVM, boosted trees) produce poorly
    > calibrated probabilities

    -   High-confidence predictions may be wrong; low-confidence may be
        > right

-   **Use cases**: Threshold-based decisions, ranking, combining with
    > other models

**Calibration Methods**

-   **Platt Scaling** (method=\'sigmoid\'):

    -   Fits logistic regression to decision function → probabilities

    -   Works well for SVM, requires small validation set

-   **Isotonic Regression** (method=\'isotonic\'):

    -   Non-parametric; learns monotonic mapping

    -   More flexible but needs more data; risk of overfitting

-   **Implementation**:\
    > \
    > python

from sklearn.calibration import CalibratedClassifierCV

calibrated_svc = CalibratedClassifierCV(LinearSVC(), method=\'sigmoid\',
cv=5)

calibrated_svc.fit(X_train, y_train)

probs = calibrated_svc.predict_proba(X_test)

-   

**Calibration Evaluation**

-   **Reliability diagram**: 10-15 bins; plot mean predicted prob vs
    > empirical accuracy

-   **Expected Calibration Error (ECE)**: Weighted avg of \|predicted -
    > empirical\| per bin

-   **Brier score**: MSE of probabilities; lower is better

-   **Log-loss**: Cross-entropy; penalizes overconfident wrong
    > predictions

## **5.4 Ensemble Methods**

**Motivation**: Combine diverse models to reduce variance, improve
robustness

**1. Voting Ensembles** (Simple, Effective)

-   **Hard Voting**: Majority vote on predicted classes

    -   Use when models have similar calibration

    -   VotingClassifier(estimators, voting=\'hard\')

-   **Soft Voting**: Average predicted probabilities → argmax

    -   Usually better; requires predict_proba

    -   VotingClassifier(estimators, voting=\'soft\')

-   **Weighted Voting**: Assign weights proportional to validation
    > macro-F1

    -   Manually tune weights or optimize on validation set

    -   VotingClassifier(estimators, voting=\'soft\', weights=\[0.4,
        > 0.3, 0.3\])

**Example Ensemble**:

python

from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(\[

(\'lr\', LogisticRegression(C=10)),

(\'svm\', CalibratedClassifierCV(LinearSVC(C=1.0))),

(\'nb\', ComplementNB(alpha=0.5)),

\], voting=\'soft\', weights=\[0.4, 0.5, 0.1\])

**2. Stacking** (Meta-Learning)

-   **Architecture**:

    -   **Base layer**: 3-5 diverse models (e.g., LinearSVC,
        > LogisticReg, RandomForest, XGBoost, ComplementNB)

    -   **Meta-learner**: Logistic Regression, Ridge, or simple linear
        > model trained on base predictions

-   **Training**:

    -   Generate OOF predictions from base models via CV

    -   Train meta-learner on OOF predictions (features) and true labels
        > (target)

    -   For test: Base models predict on test → meta-learner predicts
        > final output

-   **Implementation**:\
    > \
    > python

from sklearn.ensemble import StackingClassifier

stacker = StackingClassifier(

estimators=\[

(\'svm\', LinearSVC()),

(\'lr\', LogisticRegression()),

(\'rf\', RandomForestClassifier()),

\],

final_estimator=LogisticRegression(),

cv=5, *\# for OOF generation*

)

stacker.fit(X_train, y_train)

-   

-   **Advantage**: Learns optimal way to combine models; often 1-2% gain
    > over voting

-   **Risk**: Overfitting if base models are too similar or meta-learner
    > too complex

**3. Bagging** (Already in Random Forest)

-   **Custom Bagging**: Apply to other models (e.g., Logistic
    > Regression)\
    > \
    > python

from sklearn.ensemble import BaggingClassifier

bagging_lr = BaggingClassifier(LogisticRegression(), n_estimators=10,
max_samples=0.8)

-   

-   **Purpose**: Reduce variance via bootstrap aggregation

-   **Use case**: Less common for text; mostly for tree-based models

**4. Snapshot Ensembles** (Deep Learning Specific, Covered in Step 6)

-   Save model at multiple epochs during cyclic learning rate schedule

-   Average predictions from snapshots

**5. Model Selection Ensembles**

-   **Dynamic selection**: Choose best model per test instance based on
    > k-NN of training instances

-   **Complex**: Rarely used; interesting research direction

**Ensemble Evaluation**

-   Compare ensemble to best individual model via 10-fold CV

-   Statistical test: Is improvement significant? (Paired t-test,
    > Wilcoxon)

-   Visualize: Ensemble vs individual confusion matrices, per-class F1

## **5.5 Error Analysis & Iterative Refinement**

**Deep Dive into Misclassifications**

-   **Load OOF predictions** from best models

-   **Identify confusion patterns**:

    -   Top 5 confused category pairs (e.g., A → B happens 50 times)

    -   Symmetric vs asymmetric confusions

-   **Case studies**: Read 10-20 misclassified documents per confusion
    > pair

    -   **Look for**:

        -   Shared vocabulary, ambiguous content

        -   Mislabeling in ground truth

        -   Very short or very long documents

        -   Heavy quoting or signatures dominating

**Feature Engineering Guided by Errors**

-   **Example findings**:

    -   comp.sys.mac ↔ comp.sys.ibm: Add char n-grams to capture
        > \"Mac\", \"IBM\", \"Apple\" stylistically

    -   sci.electronics ↔ sci.crypt: Add keyword features for
        > \"encryption\", \"circuit\"

    -   talk.politics.\* confusions: High vocabulary overlap; may need
        > more context (sentence embeddings?)

**Model Adjustment**

-   **Class-specific weighting**: If one class performs poorly, try
    > class_weight={class_id: weight}

-   **Per-class thresholds**: Adjust decision thresholds independently
    > (e.g., increase threshold for overconfident class)

-   **Hard negative mining**: Augment training with near-boundary
    > examples

**Iterative Loop**

1.  Train model → Collect OOF predictions

2.  Analyze errors → Identify patterns

3.  Engineer features or adjust model → Retrain

4.  Measure improvement → Repeat if gain \>0.5% macro-F1

## **5.6 Deliverables**

-   **Optimized feature sets**: Best TF-IDF config, hybrid combinations,
    > selected features

-   **Tuned models**: Top 3-5 models with optimal hyperparameters, saved
    > via joblib

-   **OOF predictions**: All models, for ensemble training

-   **Hyperparameter search logs**: Best params, validation curves,
    > search history

-   **Ensemble models**: Voting, stacking with validation performance

-   **Comparison report**:

    -   Table: Model \| Baseline Macro-F1 \| Tuned Macro-F1 \| Gain \|
        > Training Time

    -   Confusion matrices: Before/after tuning

    -   Per-class F1 improvements

-   **Error analysis document**: Key confusion pairs, hypotheses,
    > remediation attempts

## **Step 6: Advanced Modeling with Deep Learning (PyTorch & TensorFlow)**

## **6.1 Deep Learning Setup & Data Preparation**

**Environment Configuration**

-   **Libraries**: PyTorch ≥2.0, TensorFlow ≥2.13, transformers (Hugging
    > Face), torchtext (optional)

-   **Hardware**: GPU strongly recommended (NVIDIA with CUDA 11.8+,
    > 12GB+ VRAM for transformers)

-   **Mixed precision**: Enable for faster training (PyTorch AMP,
    > TensorFlow mixed_float16)

**Text Preprocessing for Deep Learning**

-   **Tokenization**:

    -   Word-level: Build vocabulary from training set (min_freq=2-5)

    -   Subword: Use Hugging Face tokenizers (BPE, WordPiece) for
        > transformers

    -   Special tokens: \<PAD\>, \<UNK\>, \<CLS\>, \<SEP\> (BERT-style)

-   **Sequence handling**:

    -   **Max length**: 256-512 tokens (balance between coverage and
        > compute)

    -   **Padding**: Post-padding (add \<PAD\> at end) or pre-padding

    -   **Truncation**: Truncate documents exceeding max_length; log
        > truncation statistics

-   **Vocabulary**:

    -   Size: 20k-50k tokens (covers \>95% of corpus)

    -   OOV handling: Map rare words to \<UNK\>; FastText embeddings
        > help

**Data Loaders**

-   **PyTorch**:\
    > \
    > python

from torch.utils.data import Dataset, DataLoader

class NewsDataset(Dataset):

def \_\_init\_\_(self, texts, labels, tokenizer, max_len):

self.encodings = tokenizer(texts, truncation=True,
padding=\'max_length\', max_length=max_len)

self.labels = labels

def \_\_getitem\_\_(self, idx):

item = {k: torch.tensor(v\[idx\]) for k, v in self.encodings.items()}

item\[\'labels\'\] = torch.tensor(self.labels\[idx\])

return item

def \_\_len\_\_(self):

return len(self.labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

-   

-   **TensorFlow**:

    -   tf.data.Dataset.from_tensor_slices() with .batch(), .prefetch()

    -   Tokenize and pad in preprocessing; or use TextVectorization
        > layer

**Train/Validation/Test Splits**

-   **10-fold CV for deep learning**: Expensive but rigorous

    -   Train separate model per fold; aggregate OOF predictions

    -   Save best checkpoint per fold

-   **Alternative (faster)**: Single train/validation/test split

    -   Use official test set as final holdout

    -   80/20 train/validation from training set for hyperparameter
        > tuning

    -   Report results on test set only once

## **6.2 PyTorch Model Architectures**

**1. Text CNN (Convolutional Neural Network)**

**Architecture**:

python

import torch.nn as nn

class TextCNN(nn.Module):

def \_\_init\_\_(self, vocab_size, embed_dim, num_classes, num_filters,
filter_sizes, dropout=0.5):

super().\_\_init\_\_()

self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

self.convs = nn.ModuleList(\[

nn.Conv1d(embed_dim, num_filters, kernel_size=fs)

for fs in filter_sizes

\])

self.fc = nn.Linear(num_filters \* len(filter_sizes), num_classes)

self.dropout = nn.Dropout(dropout)

def forward(self, x):

*\# x: \[batch, seq_len\]*

embedded = self.embedding(x).permute(0, 2, 1) *\# \[batch, embed_dim,
seq_len\]*

conved = \[torch.relu(conv(embedded)) for conv in self.convs\] *\#
\[batch, num_filters, seq_len-fs+1\]*

pooled = \[torch.max(conv, dim=2)\[0\] for conv in conved\] *\# \[batch,
num_filters\]*

cat = torch.cat(pooled, dim=1) *\# \[batch, num_filters \*
len(filter_sizes)\]*

output = self.fc(self.dropout(cat))

return output

**Hyperparameters**:

-   embed_dim: 100, 200, 300 (match pretrained embeddings if used)

-   num_filters: 100, 128, 256 per filter size

-   filter_sizes: (trigrams, 4-grams, 5-grams) or

-   dropout: 0.3, 0.5

-   **Embeddings**: Initialize randomly, or load GloVe/Word2Vec; freeze
    > or fine-tune

**Training**:

-   Loss: nn.CrossEntropyLoss()

-   Optimizer: Adam (lr=1e-3, 5e-4), AdamW with weight decay

-   Batch size: 32, 64, 128

-   Epochs: 10-30 with early stopping (patience=5)

-   Learning rate schedule: ReduceLROnPlateau or CosineAnnealing

-   Gradient clipping:
    > torch.nn.utils.clip_grad_norm\_(model.parameters(), max_norm=1.0)

**Expected Performance**: 0.84-0.88 macro-F1 (competitive with best
classical models)

**2. Bidirectional LSTM/GRU with Attention**

**Architecture**:

python

class BiLSTMAttention(nn.Module):

def \_\_init\_\_(self, vocab_size, embed_dim, hidden_dim, num_classes,
num_layers=2, dropout=0.5):

super().\_\_init\_\_()

self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,

bidirectional=True, dropout=dropout if num_layers \> 1 else 0,
batch_first=True)

self.attention = nn.Linear(hidden_dim \* 2, 1)

self.fc = nn.Linear(hidden_dim \* 2, num_classes)

self.dropout = nn.Dropout(dropout)

def forward(self, x):

*\# x: \[batch, seq_len\]*

embedded = self.embedding(x) *\# \[batch, seq_len, embed_dim\]*

lstm_out, \_ = self.lstm(embedded) *\# \[batch, seq_len,
hidden_dim\*2\]*

*\# Attention*

attn_weights = torch.softmax(self.attention(lstm_out), dim=1) *\#
\[batch, seq_len, 1\]*

attn_out = torch.sum(attn_weights \* lstm_out, dim=1) *\# \[batch,
hidden_dim\*2\]*

output = self.fc(self.dropout(attn_out))

return output

**Hyperparameters**:

-   hidden_dim: 128, 256

-   num_layers: 1, 2, 3

-   dropout: 0.3, 0.5

-   **Attention**: Can use multi-head attention (nn.MultiheadAttention)
    > for more capacity

**Training**: Similar to CNN; LSTMs often need lower learning rates
(1e-3 to 1e-4)

**Expected Performance**: 0.85-0.89 macro-F1; captures longer
dependencies than CNN

**3. Simple MLP Baseline (Sanity Check)**

-   Input: Pre-computed document embeddings (e.g., TF-IDF weighted GloVe
    > average)

-   Architecture: Linear → ReLU → Dropout → Linear → ReLU → Dropout →
    > Linear (output)

-   Fast to train; baseline for more complex architectures

**4. Hybrid CNN-LSTM**

python

class CNNLSTM(nn.Module):

def \_\_init\_\_(self, vocab_size, embed_dim, num_filters, filter_sizes,
lstm_hidden, num_classes, dropout=0.5):

super().\_\_init\_\_()

self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

self.convs = nn.ModuleList(\[nn.Conv1d(embed_dim, num_filters, fs) for
fs in filter_sizes\])

self.lstm = nn.LSTM(num_filters \* len(filter_sizes), lstm_hidden,
batch_first=True, bidirectional=True)

self.fc = nn.Linear(lstm_hidden \* 2, num_classes)

self.dropout = nn.Dropout(dropout)

def forward(self, x):

embedded = self.embedding(x).permute(0, 2, 1)

conved = \[torch.relu(conv(embedded)) for conv in self.convs\]

pooled = \[conv.permute(0, 2, 1) for conv in conved\] *\# \[batch,
seq_len\', num_filters\]*

cat = torch.cat(pooled, dim=2) *\# \[batch, seq_len\', num_filters \*
len(filter_sizes)\]*

lstm_out, \_ = self.lstm(cat)

output = self.fc(self.dropout(lstm_out\[:, -1, :\])) *\# Use last hidden
state*

return output

-   **Rationale**: CNN extracts local n-gram features; LSTM models
    > sequential dependencies

-   **Expected Performance**: Marginal gain over CNN alone (0.86-0.90
    > macro-F1)

**PyTorch Training Best Practices**

-   **Mixed precision** (if GPU supports):\
    > \
    > python

from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:

with autocast():

output = model(batch\[\'input_ids\'\])

loss = criterion(output, batch\[\'labels\'\])

scaler.scale(loss).backward()

scaler.step(optimizer)

scaler.update()

-   

-   **Gradient clipping**: Essential for RNNs to prevent exploding
    > gradients

-   **Early stopping**: Monitor validation loss or macro-F1; stop if no
    > improvement for 5-10 epochs

-   **Model checkpointing**: Save best model based on validation metric

-   **Learning rate scheduling**:

    -   ReduceLROnPlateau: Reduce LR when validation metric plateaus

    -   CosineAnnealingLR: Smooth decay with restarts

## **6.3 TensorFlow/Keras Model Architectures**

**1. Dense Neural Network (Baseline)**

python

from tensorflow.keras import layers, models

def build_dnn(vocab_size, embed_dim, num_classes, max_len):

model = models.Sequential(\[

layers.Embedding(vocab_size, embed_dim, input_length=max_len),

layers.GlobalAveragePooling1D(), *\# Or Flatten*

layers.Dense(512, activation=\'relu\'),

layers.Dropout(0.5),

layers.Dense(256, activation=\'relu\'),

layers.Dropout(0.5),

layers.Dense(num_classes, activation=\'softmax\')

\])

return model

model = build_dnn(vocab_size=50000, embed_dim=300, num_classes=4,
max_len=512)

model.compile(optimizer=\'adam\',
loss=\'sparse_categorical_crossentropy\', metrics=\[\'accuracy\'\])

**2. CNN for Text (Keras Functional API)**

python

def build_text_cnn(vocab_size, embed_dim, num_classes, max_len,
num_filters=128, filter_sizes=\[3,4,5\], dropout=0.5):

inputs = layers.Input(shape=(max_len,))

embedding = layers.Embedding(vocab_size, embed_dim)(inputs)

conv_blocks = \[\]

for fs in filter_sizes:

conv = layers.Conv1D(num_filters, fs, activation=\'relu\')(embedding)

pool = layers.GlobalMaxPooling1D()(conv)

conv_blocks.append(pool)

concat = layers.Concatenate()(conv_blocks) if len(conv_blocks) \> 1 else
conv_blocks\[0\]

dropout_layer = layers.Dropout(dropout)(concat)

outputs = layers.Dense(num_classes,
activation=\'softmax\')(dropout_layer)

model = models.Model(inputs, outputs)

return model

model = build_text_cnn(vocab_size=50000, embed_dim=300, num_classes=4,
max_len=512)

model.compile(optimizer=\'adam\',
loss=\'sparse_categorical_crossentropy\', metrics=\[\'accuracy\'\])

**3. Bidirectional LSTM**

python

def build_bilstm(vocab_size, embed_dim, num_classes, max_len,
lstm_units=128, dropout=0.5):

model = models.Sequential(\[

layers.Embedding(vocab_size, embed_dim, input_length=max_len),

layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True)),

layers.GlobalMaxPooling1D(),

layers.Dense(128, activation=\'relu\'),

layers.Dropout(dropout),

layers.Dense(num_classes, activation=\'softmax\')

\])

return model

model = build_bilstm(vocab_size=50000, embed_dim=300, num_classes=4,
max_len=512)

model.compile(optimizer=\'adam\',
loss=\'sparse_categorical_crossentropy\', metrics=\[\'accuracy\'\])

**4. LSTM with Attention (Custom Layer)**

python

class Attention(layers.Layer):

def \_\_init\_\_(self):

super().\_\_init\_\_()

def build(self, input_shape):

self.W = self.add_weight(shape=(input_shape\[-1\], 1),
initializer=\'random_normal\', trainable=True)

def call(self, inputs):

attn_weights = tf.nn.softmax(tf.matmul(inputs, self.W), axis=1)

attn_out = tf.reduce_sum(attn_weights \* inputs, axis=1)

return attn_out

def build_lstm_attention(vocab_size, embed_dim, num_classes, max_len,
lstm_units=128, dropout=0.5):

inputs = layers.Input(shape=(max_len,))

embedding = layers.Embedding(vocab_size, embed_dim)(inputs)

lstm = layers.Bidirectional(layers.LSTM(lstm_units,
return_sequences=True))(embedding)

attention = Attention()(lstm)

dropout_layer = layers.Dropout(dropout)(attention)

outputs = layers.Dense(num_classes,
activation=\'softmax\')(dropout_layer)

model = models.Model(inputs, outputs)

return model

**Training Configuration**

python

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,
ReduceLROnPlateau

callbacks = \[

EarlyStopping(monitor=\'val_loss\', patience=5,
restore_best_weights=True),

ModelCheckpoint(\'best_model.h5\', monitor=\'val_accuracy\',
save_best_only=True),

ReduceLROnPlateau(monitor=\'val_loss\', factor=0.5, patience=3,
min_lr=1e-7)

\]

history = model.fit(

train_dataset,

validation_data=val_dataset,

epochs=30,

callbacks=callbacks,

verbose=1

)

## **6.4 Transfer Learning with Transformers**

**Hugging Face Transformers Integration**

**1. BERT Fine-Tuning (TensorFlow)**

python

from transformers import TFBertForSequenceClassification, BertTokenizer

tokenizer = BertTokenizer.from_pretrained(\'bert-base-uncased\')

model =
TFBertForSequenceClassification.from_pretrained(\'bert-base-uncased\',
num_labels=4)

*\# Tokenize*

train_encodings = tokenizer(train_texts, truncation=True,
padding=\'max_length\', max_length=256, return_tensors=\'tf\')

train_dataset =
tf.data.Dataset.from_tensor_slices((dict(train_encodings),
train_labels)).batch(16)

*\# Compile*

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=\[\'accuracy\'\])

*\# Train*

model.fit(train_dataset, validation_data=val_dataset, epochs=3,
callbacks=callbacks)

**2. DistilBERT (Faster, Lighter)**

-   Same API; replace bert-base-uncased with distilbert-base-uncased

-   40% smaller, 60% faster, retains 97% of BERT performance

-   **Recommended** for 20NG unless you need absolute best accuracy

**3. RoBERTa (Robustly Optimized BERT)**

-   roberta-base or roberta-large

-   Often 1-2% better than BERT; trained longer with more data

**4. PyTorch Transformers**

python

from transformers import BertTokenizer, BertForSequenceClassification,
Trainer, TrainingArguments

tokenizer = BertTokenizer.from_pretrained(\'bert-base-uncased\')

model =
BertForSequenceClassification.from_pretrained(\'bert-base-uncased\',
num_labels=20)

train_encodings = tokenizer(train_texts, truncation=True,
padding=\'max_length\', max_length=256)

train_dataset = NewsDataset(train_encodings, train_labels)

training_args = TrainingArguments(

output_dir=\'./results\',

num_train_epochs=3,

per_device_train_batch_size=16,

per_device_eval_batch_size=32,

warmup_steps=500,

weight_decay=0.01,

logging_dir=\'./logs\',

logging_steps=100,

evaluation_strategy=\'epoch\',

save_strategy=\'epoch\',

load_best_model_at_end=True,

metric_for_best_model=\'f1\',

)

trainer = Trainer(

model=model,

args=training_args,

train_dataset=train_dataset,

eval_dataset=val_dataset,

compute_metrics=compute_metrics_fn, *\# Custom function returning
{\'f1\': macro_f1, \'accuracy\': acc}*

)

trainer.train()

**Fine-Tuning Strategies**

-   **Freeze embeddings**: First epoch, only train classification head\
    > \
    > python

for param in model.bert.embeddings.parameters():

param.requires_grad = False

-   

-   Unfreeze after first epoch for full fine-tuning

-   **Layer-wise learning rates** (discriminative fine-tuning):

    -   Lower LR for lower layers, higher for upper layers and head

    -   Implementation: PyTorch optimizer with parameter groups

-   **Gradual unfreezing**: Unfreeze layers progressively from top to
    > bottom

**Hyperparameters for Transformers**

-   **Learning rate**: 2e-5, 3e-5, 5e-5 (much lower than from-scratch
    > models)

-   **Batch size**: 8, 16, 32 (limited by VRAM; use gradient
    > accumulation if needed)

-   **Epochs**: 2-5 (transformers overfit quickly on small datasets)

-   **Max length**: 128, 256, 512 (longer = more memory, slower; 256 is
    > good balance for 20NG)

-   **Warmup steps**: 500, 1000 (linear warmup of LR from 0)

-   **Weight decay**: 0.01, 0.1 (L2 regularization)

**Expected Performance**: 0.88-0.93 macro-F1 (state-of-the-art for 20NG)

## **6.5 Advanced Training Techniques**

**Data Augmentation for Text** (Use Sparingly)

-   **EDA (Easy Data Augmentation)**:

    -   Synonym replacement (WordNet)

    -   Random insertion/deletion/swap of words

    -   **Caution**: May introduce noise; validate impact on 20NG

-   **Back-translation**: Translate to another language and back
    > (expensive, small gain)

-   **Contextual word substitution**: Use BERT to replace words with
    > context-appropriate alternatives

-   **Not recommended for 20NG**: Dataset is large enough; augmentation
    > rarely helps

**Regularization Beyond Dropout**

-   **Label smoothing**: Soften one-hot labels (e.g., 0.9 for correct
    > class, 0.1/19 for others)\
    > \
    > python

loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

-   

-   **Mixup**: Interpolate between training examples (works for text
    > embeddings)

-   **Cutoff**: Randomly zero out embedding dimensions (similar to
    > dropout)

**Optimization Techniques**

-   **Gradient accumulation**: Simulate larger batch size when VRAM is
    > limited\
    > \
    > python

*\# PyTorch*

accumulation_steps = 4

for i, batch in enumerate(train_loader):

loss = model(batch) / accumulation_steps

loss.backward()

if (i+1) % accumulation_steps == 0:

optimizer.step()

optimizer.zero_grad()

-   

-   **Learning rate warm-up**: Linear increase from 0 to target LR over
    > first 5-10% of training

-   **Cyclic learning rates**: Oscillate LR between bounds (snapshot
    > ensembles benefit from this)

**Multi-Task Learning** (Advanced, Optional)

-   **Auxiliary tasks**: Predict document length category, sentiment,
    > contains code snippet

-   **Architecture**: Shared encoder + multiple task-specific heads

-   **Benefit**: Regularization via inductive bias from related tasks

-   **Effort**: High; justify with ablation study

## **6.6 Deep Learning Evaluation & Interpretation**

**Metrics & Diagnostics**

-   **Training curves**: Plot loss and macro-F1 vs epoch for train and
    > validation

    -   Check for: Overfitting (train high, val plateaus), underfitting
        > (both low)

-   **Per-class metrics**: Precision, recall, F1 for each of 20
    > categories

-   **Confusion matrix**: Heatmap on test set; compare to classical
    > models

-   **Calibration**: Reliability diagram for transformer models

**Model Interpretability**

**1. Attention Visualization**

-   **For attention-based models (LSTM+Attention, Transformers)**:

    -   Extract attention weights for a document

    -   Highlight words with high attention (HTML visualization or
        > heatmap)

-   python

*\# Example with Transformers*

outputs = model(\*\*inputs, output_attentions=True)

attentions = outputs.attentions *\# Tuple of layers, each \[batch,
heads, seq_len, seq_len\]*

*\# Average across heads and layers; visualize as heatmap over tokens*

-   

**2. Saliency Maps (Gradient-Based)**

-   **Compute gradients of output w.r.t. input embeddings\
    > **

-   **Interpretation**: High gradient magnitude = important word for
    > prediction\
    > \
    > python

*\# PyTorch*

model.eval()

embeddings = model.embedding(input_ids)

embeddings.retain_grad()

output = model(embeddings)

output\[0, predicted_class\].backward()

saliency = embeddings.grad.abs().sum(dim=2) *\# \[batch, seq_len\]*

-   

**3. LIME (Local Interpretable Model-Agnostic Explanations)**

-   **Process**: Perturb input text (mask words), observe prediction
    > changes

-   **Output**: Words that most influenced the prediction\
    > \
    > python

from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=class_names)

exp = explainer.explain_instance(text, model.predict_proba,
num_features=10)

exp.show_in_notebook()

-   

**4. SHAP (SHapley Additive exPlanations)**

-   **Process**: Compute Shapley values for each word/token

-   **Output**: Feature importance with directional impact
    > (positive/negative)\
    > \
    > python

import shap

explainer = shap.Explainer(model.predict, tokenizer)

shap_values = explainer(\[text\])

shap.plots.text(shap_values)

-   

**5. Embedding Space Visualization**

-   **Extract**: Final hidden states or \[CLS\] token embeddings for all
    > documents

-   **Reduce**: t-SNE or UMAP to 2D

-   **Plot**: Scatter colored by category; assess cluster quality

-   **Interactive**: Plotly with hover showing document text

**Adversarial Testing** (Optional)

-   **Robustness check**: Perturb text (typos, synonym replacement) and
    > measure prediction stability

-   **Tools**: TextAttack library for adversarial attacks on text models

## **6.7 Ensemble Deep + Classical Models**

-   **Heterogeneous stacking**: Base models = \[LinearSVC, LogisticReg,
    > BERT\], Meta-learner = LogisticReg

-   **Probability averaging**: Average predicted probabilities from top
    > classical and top deep model

-   **Weighted ensemble**: Optimize weights on validation set to
    > maximize macro-F1

-   **Expected gain**: 0.5-1.5% over best individual model

## **6.8 Deliverables**

-   **Trained models**: Saved checkpoints (PyTorch .pth, TensorFlow .h5,
    > Transformers save_pretrained())

-   **Training logs**: Loss/accuracy curves, TensorBoard logs, W&B
    > dashboards

-   **OOF predictions**: From 10-fold CV (if performed) or validation
    > set

-   **Interpretation visualizations**: Attention heatmaps, saliency
    > maps, LIME/SHAP explanations for 10-20 sample documents

-   **Performance comparison**: Deep models vs classical models (table +
    > bar chart)

-   **Code**: Modular training scripts, reusable model classes,
    > inference pipeline

-   **Report**: Architecture decisions, hyperparameter tuning process,
    > ablation studies (e.g., effect of attention, embedding
    > initialization)

## **Step 7: Final Evaluation, Model Selection, Ensembling & Deployment Preparation**

    ## **7.1 Comprehensive Model Comparison**

    **Master Performance Table**

    ----------------------------------------------------------------------------------------------------------------------------------
  **Model        **Model**            **Features**   **Macro-F1   **Std   **Test       **Test   **Training   **Inference   **Model
  Family**                                           (CV)**       Dev**   Macro-F1**   Acc**    Time**       (ms/doc)**    Size
                                                                                                                           (MB)**
  -------------- -------------------- -------------- ------------ ------- ------------ -------- ------------ ------------- ---------
  Baseline       DummyClassifier      \-             0.050        0.005   0.050        0.050    \<1s         \<1ms         \<1MB

  Classical      MultinomialNB        TF-IDF (1,2)   0.XXX        0.XXX   0.XXX        0.XXX    Xs           Xms           XMB

  Classical      ComplementNB         TF-IDF (1,2)   0.XXX        0.XXX   0.XXX        0.XXX    Xs           Xms           XMB

  Classical      LogisticRegression   Word+Char      0.XXX        0.XXX   0.XXX        0.XXX    XXs          Xms           XMB
                                      TF-IDF                                                                               

  Classical      LinearSVC            Word+Char      0.XXX        0.XXX   0.XXX        0.XXX    XXs          Xms           XMB
                                      TF-IDF                                                                               

  Ensemble       Voting (LR+SVC+NB)   Word+Char      0.XXX        0.XXX   0.XXX        0.XXX    XXs          Xms           XMB
  (Classical)                         TF-IDF                                                                               

  Ensemble       Stacking             Multiple       0.XXX        0.XXX   0.XXX        0.XXX    XXXs         XXms          XMB
  (Classical)                                                                                                              

  Tree-Based     XGBoost              TF-IDF + Aux   0.XXX        0.XXX   0.XXX        0.XXX    XXXs         XXms          XXX MB

  Deep Learning  TextCNN              Word Emb       0.XXX        0.XXX   0.XXX        0.XXX    XXXs         XXms          XXX MB
  (PyTorch)                           (GloVe)                                                                              

  Deep Learning  BiLSTM+Attention     Word Emb       0.XXX        0.XXX   0.XXX        0.XXX    XXXXs        XXms          XXX MB
  (PyTorch)                                                                                                                

  Deep Learning  CNN                  Word Emb       0.XXX        0.XXX   0.XXX        0.XXX    XXXs         XXms          XXX MB
  (TF)                                                                                                                     

  Transformers   DistilBERT           Subword        0.XXX        0.XXX   0.XXX        0.XXX    XXXXs        XXXms         XXX MB
  (TF)                                                                                                                     

  Transformers   BERT-base            Subword        0.XXX        0.XXX   0.XXX        0.XXX    XXXXXs       XXXms         XXX MB
  (PyTorch)                                                                                                                

  Transformers   RoBERTa-base         Subword        0.XXX        0.XXX   0.XXX        0.XXX    XXXXXs       XXXms         XXX MB
  (PyTorch)                                                                                                                

  Ensemble       Stacking (SVC+BERT)  Multiple       0.XXX        0.XXX   0.XXX        0.XXX    XXXXXs       XXXms         XXX MB
  (Hybrid)                                                                                                                 
  ----------------------------------------------------------------------------------------------------------------------------------

**Visualization Suite**

1.  **Grouped bar chart**: Macro-F1 comparison (all models, sorted by
    > performance)

2.  **Scatter plot**: Accuracy vs training time, accuracy vs inference
    > latency, accuracy vs model size

3.  **Box plots**: CV fold macro-F1 distributions for top 5 models

4.  **Confusion matrices**: Side-by-side heatmaps for top 3 models
    > (classical, deep, transformer)

5.  **Per-class F1 radar chart**: 20-sided polygon comparing 3-5 best
    > models

6.  **ROC curves**: Multiclass one-vs-rest for best model (20 curves,
    > averaged)

7.  **Precision-Recall curves**: Similar to ROC

8.  **Calibration plots**: Reliability diagrams for probabilistic models
    > (LR, BERT)

## **7.2 Statistical Significance & Effect Size**

**Pairwise Comparisons** (Best Classical vs Best Deep vs Best Ensemble)

-   **Paired t-test** or **Wilcoxon signed-rank test**: Compare
    > fold-wise or bootstrap macro-F1

    -   H0: No difference; reject if p \< 0.05

-   **McNemar\'s test**: Compare predictions on test set
    > (correct/incorrect contingency)

-   **Cohen\'s d**: Measure effect size; is 1% gain practically
    > significant?

-   **Confidence intervals**: Bootstrap 95% CI for macro-F1 difference

**Multiple Comparison Correction**

-   **Bonferroni correction**: Adjust α for multiple tests

-   **Friedman + Nemenyi**: Rank models across folds; identify
    > significantly different groups

## **7.3 Error Analysis on Test Set**

**Load Test Predictions**

-   Best classical model (e.g., LinearSVC)

-   Best deep model (e.g., BiLSTM+Attention or DistilBERT)

-   Best ensemble

**Confusion Analysis**

-   **Aggregate confusion matrix** for each model

-   **Identify persistent confusions**: Category pairs confused by all
    > models

    -   Example: talk.politics.mideast ↔ talk.politics.misc (similar
        > topics)

-   **Model-specific errors**: Confusions unique to one model type

    -   Example: Deep models may confuse sci.crypt ↔ comp.sys.\*
        > (technical jargon overlap); classical models may not

**Case Studies**

-   **Consistently correct**: Documents predicted correctly by all
    > models (easy cases)

-   **Consistently wrong**: Misclassified by all models

    -   Manual review: Are they mislabeled? Ambiguous content?
        > Short/truncated?

-   **Model disagreements**: Where models differ; which is correct?

    -   Hypothesis: Ensemble of diverse models can resolve these via
        > voting

**Categorize Errors**

-   **Document characteristics**:

    -   Very short (\<50 words)

    -   Very long (\>1000 words, may be truncated by deep models)

    -   High quote ratio (dominated by quoted text)

    -   Ambiguous topic (spans multiple categories)

-   **Linguistic patterns**:

    -   Rare vocabulary (OOV issues for pretrained embeddings)

    -   Sarcasm/irony (difficult for all models)

    -   Code-heavy (formatting issues)

## **7.4 Final Model Selection**

**Selection Criteria**

1.  **Performance**: Macro-F1 (primary), accuracy, per-class F1 balance

2.  **Inference speed**: Latency requirement for production (e.g.,
    > \<100ms)

3.  **Model size**: Memory/storage constraints (e.g., \<500MB for edge
    > deployment)

4.  **Interpretability**: Stakeholder need for explainability (linear
    > models win)

5.  **Training cost**: Can model be retrained easily? (classical models
    > easier than transformers)

6.  **Robustness**: Performance on perturbed inputs, out-of-distribution
    > data

**Decision Matrix Example**

  ---------------------------------------------------------------------------------------
  **Criterion**      **Weight**   **LinearSVC**   **DistilBERT**   **Ensemble
                                                                   (SVC+BERT)**
  ------------------ ------------ --------------- ---------------- ----------------------
  Macro-F1           40%          0.880           0.910            0.920

  Inference Speed    25%          10ms            250ms            260ms

  Model Size         15%          50MB            250MB            300MB

  Interpretability   10%          High            Low              Medium

  Retraining Cost    10%          Low             High             High

  **Weighted Score**              **X.XX**        **X.XX**         **X.XX**
  ---------------------------------------------------------------------------------------

**Recommendation Scenarios**

-   **Production (speed critical)**: LinearSVC or LogisticReg with
    > word+char TF-IDF

-   **Best accuracy (research)**: BERT or ensemble

-   **Balanced (real-world)**: DistilBERT or stacking ensemble with fast
    > classical + lightweight transformer

## **7.5 Final Holdout Test Set Evaluation**

**One-Time Evaluation** (No Tuning After This)

-   Load best model(s) selected above

-   Predict on official test set (7,532 samples)

-   **Metrics**:

    -   Macro-F1, weighted-F1, accuracy

    -   Per-class precision, recall, F1

    -   Confusion matrix (20x20)

    -   Classification report (sklearn)

    -   ROC-AUC (one-vs-rest), PR-AUC

    -   Calibration metrics (if probabilistic)

-   **Confidence analysis**:

    -   Distribution of predicted probabilities (histogram)

    -   Confidence vs correctness (scatter: high confidence but wrong =
        > overconfident)

**Comparison to Literature**

-   Expected performance on 20NG (clean, headers removed):

    -   Naive Bayes: 0.77-0.83

    -   Linear SVM: 0.84-0.90

    -   CNN/LSTM: 0.85-0.89

    -   BERT/RoBERTa: 0.88-0.93

-   **Report**: Where does your model stand? Match expectations?
    > Outperform?

## **7.6 Model Interpretability & Insights**

**Global Interpretability (Model-Level Insights)**

-   **Linear models (LogReg, SVM)**:

    -   Extract top positive/negative coefficient words per class

    -   Visualize: Heatmap of top 20 words × 20 classes

    -   Word clouds: Top words per category by coefficient magnitude

-   **Tree-based models (RF, XGBoost)**:

    -   Feature importance: Gini importance, permutation importance

    -   Partial dependence plots: Effect of top features on predictions

-   **Deep models**:

    -   Embedding space: t-SNE/UMAP of learned embeddings; color by
        > category

    -   Attention aggregation: Average attention weights across
        > documents per class

**Local Interpretability (Instance-Level Explanations)**

-   **LIME**: Explain 20-30 test predictions (both correct and
    > incorrect)

    -   Show: Top words contributing to predicted class

    -   Compare: True label vs predicted label explanation

-   **SHAP**: Waterfall plots for individual predictions

    -   Quantify: Each word\'s contribution to final logit/probability

-   **Attention heatmaps**: For transformer/LSTM+attention models

    -   Highlight: Words with high attention scores

    -   Qualitative assessment: Do highlighted words make sense?

**Insights Documentation**

-   **What makes each category distinctive?\
    > **

    -   comp.graphics: \"image\", \"graphics\", \"3d\", \"file format
        > names\"

    -   sci.crypt: \"encryption\", \"key\", \"cipher\", \"algorithm
        > names\"

    -   rec.sport.baseball: team names, player names, \"game\",
        > \"season\"

    -   talk.religion.misc: \"god\", \"believe\", \"faith\",
        > \"christian\", \"atheist\" (overlaps with alt.atheism)

-   **Why do certain confusions persist?\
    > **

    -   Vocabulary overlap (e.g., both sci.electronics and sci.crypt
        > discuss circuits/hardware)

    -   Short documents with ambiguous keywords

    -   Mislabeled ground truth (rare but possible)

-   **How do models differ in their decision-making?\
    > **

    -   Classical: Rely heavily on discriminative keywords (TF-IDF
        > weights)

    -   Deep: Capture contextual patterns, phrase compositions

    -   Transformers: Contextualized embeddings allow nuanced
        > understanding

## **7.7 Ensemble Strategies (Final)**

**Best Ensemble Configuration**

-   **Base models**: Top 3-5 models with diversity

    -   Example: LinearSVC (word+char TF-IDF), XGBoost (TF-IDF + aux
        > features), DistilBERT

-   **Combination method**:

    -   **Soft voting**: Average predicted probabilities; works if
        > models are calibrated

    -   **Weighted voting**: Optimize weights on validation set (grid
        > search or Bayesian opt)

        -   Objective: Maximize validation macro-F1

    -   **Stacking**: Train meta-learner (LogReg/Ridge) on OOF
        > predictions

-   **Implementation**:\
    > \
    > python

*\# Example: Weighted averaging*

def ensemble_predict(models, weights, X):

probas = \[model.predict_proba(X) for model in models\]

weighted_proba = np.average(probas, axis=0, weights=weights)

return np.argmax(weighted_proba, axis=1)

*\# Optimize weights on validation set*

from scipy.optimize import minimize

def objective(w):

preds = ensemble_predict(models, w, X_val)

return -f1_score(y_val, preds, average=\'macro\') *\# Negative for
minimization*

result = minimize(objective, x0=np.ones(len(models))/len(models),
bounds=\[(0,1)\]\*len(models), constraints={\'type\': \'eq\', \'fun\':
lambda w: np.sum(w) - 1})

optimal_weights = result.x

-   

-   **Evaluation**: Test macro-F1 with optimal ensemble; compare to best
    > individual model

    -   Expected gain: 0.5-1.5% (diminishing returns if models are too
        > similar)

**Snapshot Ensemble** (Deep Learning Specific)

-   Save model checkpoints at multiple epochs during cyclic learning
    > rate schedule

-   Average predictions from 5-10 snapshots

-   Benefit: Diversity without training separate models

## **7.8 Deployment Preparation**

**Model Serialization & Packaging**

-   **Classical models**:

    -   Save: joblib.dump(model, \'model.pkl\')

    -   Include: Vectorizer/scaler/feature selector pipelines

-   **Deep models (PyTorch)**:

    -   Save: torch.save(model.state_dict(), \'model.pth\') + model
        > class definition

    -   Or: TorchScript for production (torch.jit.script(model))

-   **Deep models (TensorFlow)**:

    -   Save: model.save(\'model\') (SavedModel format) or
        > model.save_weights(\'weights.h5\')

-   **Transformers**:

    -   Save: model.save_pretrained(\'model_dir\');
        > tokenizer.save_pretrained(\'model_dir\')

    -   Load: model.from_pretrained(\'model_dir\')

-   **ONNX export** (Interoperability):

    -   Convert PyTorch/TF models to ONNX for cross-framework inference

    -   Example: torch.onnx.export(model, dummy_input, \'model.onnx\')

**Inference Pipeline**

python

class NewsClassifier:

def \_\_init\_\_(self, model_path, vectorizer_path):

self.model = joblib.load(model_path)

self.vectorizer = joblib.load(vectorizer_path)

self.label_names = \[\...\] *\# Category names*

def preprocess(self, text):

*\# Apply same cleaning as training*

text = text.lower()

*\# \... (remove headers, tokenize, etc.)*

return text

def predict(self, text):

cleaned = self.preprocess(text)

features = self.vectorizer.transform(\[cleaned\])

proba = self.model.predict_proba(features)\[0\]

pred_class = np.argmax(proba)

return {

\'category\': self.label_names\[pred_class\],

\'confidence\': float(proba\[pred_class\]),

\'probabilities\': {self.label_names\[i\]: float(proba\[i\]) for i in
range(len(proba))}

}

classifier = NewsClassifier(\'best_model.pkl\', \'vectorizer.pkl\')

result = classifier.predict(\"I need help with my graphics card
driver\...\")

print(result)

*\# {\'category\': \'comp.graphics\', \'confidence\': 0.92,
\'probabilities\': {\...}}*

**API Development** (Optional)

-   **Flask REST API**:\
    > \
    > python

from flask import Flask, request, jsonify

app = Flask(\_\_name\_\_)

classifier = NewsClassifier(\'model.pkl\', \'vectorizer.pkl\')

\@app.route(\'/predict\', methods=\[\'POST\'\])

def predict():

data = request.json

text = data\[\'text\'\]

result = classifier.predict(text)

return jsonify(result)

if \_\_name\_\_ == \'\_\_main\_\_\':

app.run(host=\'0.0.0.0\', port=5000)

-   

-   **FastAPI** (Modern, Fast):\
    > \
    > python

from fastapi import FastAPI

from pydantic import BaseModel

app = FastAPI()

classifier = NewsClassifier(\'model.pkl\', \'vectorizer.pkl\')

class TextInput(BaseModel):

text: str

\@app.post(\'/predict\')

def predict(input: TextInput):

return classifier.predict(input.text)

-   

**Containerization** (Docker)

text

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install \--no-cache-dir -r requirements.txt

COPY model.pkl vectorizer.pkl app.py ./

EXPOSE 5000

CMD \[\"python\", \"app.py\"\]

**Model Compression** (Optional, for Edge Deployment)

-   **Quantization**: Reduce precision (FP32 → FP16 or INT8)

    -   PyTorch: torch.quantization.quantize_dynamic(model, {nn.Linear},
        > dtype=torch.qint8)

    -   TensorFlow Lite: Convert to TFLite with quantization

-   **Pruning**: Remove low-importance weights

-   **Distillation**: Train small model (student) to mimic large model
    > (teacher)

## **7.9 Documentation & Reproducibility**

**Code Organization**

text

20newsgroups_classification/

├── README.md \# Project overview, setup, usage

├── requirements.txt \# Python dependencies

├── environment.yml \# Conda environment (optional)

├── config/

│ ├── config.yaml \# Hyperparameters, paths, constants

│ └── logging_config.yaml \# Logging setup

├── data/

│ ├── raw/ \# Downloaded dataset cache

│ └── processed/ \# Preprocessed artifacts

├── notebooks/

│ ├── 01_data_loading_eda.ipynb

│ ├── 02_preprocessing.ipynb

│ ├── 03_baseline_models.ipynb

│ ├── 04_feature_engineering.ipynb

│ ├── 05_deep_learning_pytorch.ipynb

│ ├── 06_deep_learning_tensorflow.ipynb

│ ├── 07_transformers.ipynb

│ └── 08_ensemble_evaluation.ipynb

├── src/

│ ├── \_\_init\_\_.py

│ ├── data_loader.py \# Fetch and load data

│ ├── preprocessing.py \# Text cleaning, tokenization

│ ├── feature_engineering.py \# Vectorization, embeddings

│ ├── models/

│ │ ├── \_\_init\_\_.py

│ │ ├── classical.py \# Sklearn model wrappers

│ │ ├── pytorch_models.py \# TextCNN, BiLSTM, etc.

│ │ ├── tf_models.py \# Keras models

│ │ └── transformers.py \# BERT fine-tuning

│ ├── training/

│ │ ├── train_classical.py

│ │ ├── train_pytorch.py

│ │ ├── train_tf.py

│ │ └── train_transformers.py

│ ├── evaluation.py \# Metrics, confusion matrix, plots

│ ├── visualization.py \# EDA plots, result charts

│ ├── interpretability.py \# LIME, SHAP, attention viz

│ └── utils.py \# Helper functions

├── models/

│ └── saved_models/ \# Serialized models, vectorizers

├── results/

│ ├── figures/ \# All plots and visualizations

│ ├── metrics/ \# CSV/JSON of metrics

│ └── predictions/ \# OOF and test predictions

├── reports/

│ ├── eda_report.md

│ ├── baseline_report.md

│ ├── deep_learning_report.md

│ └── final_report.pdf \# Comprehensive final report

├── tests/

│ ├── test_preprocessing.py

│ └── test_models.py

└── app/

├── api.py \# Flask/FastAPI endpoint

├── classifier.py \# Inference class

└── Dockerfile

**README.md Structure**

text

\# 20 Newsgroups Text Classification

\## Overview

Brief description of project, dataset, and objectives.

\## Dataset

\- Source: scikit-learn \`fetch_20newsgroups\`

\- Size: 18,846 documents, 20 categories

\- Splits: Train (11,314), Test (7,532)

\## Setup

\### Requirements

\- Python 3.9+

\- See \`requirements.txt\`

\### Installation

pip install -r requirements.txt

text

\## Usage

\### Training

python src/training/train_classical.py \--config config/config.yaml

text

\### Inference

from app.classifier import NewsClassifier\
classifier = NewsClassifier(\'models/best_model.pkl\',
\'models/vectorizer.pkl\')\
result = classifier.predict(\"Your text here\...\")

text

\## Results

\| Model \| Test Macro-F1 \| Test Accuracy \|

\|\-\-\-\-\-\--\|\-\-\-\-\-\-\-\-\-\-\-\-\-\--\|\-\-\-\-\-\-\-\-\-\-\-\-\-\--\|

\| LinearSVC \| 0.880 \| 0.895 \|

\| DistilBERT \| 0.910 \| 0.921 \|

\| Ensemble \| 0.920 \| 0.928 \|

\## Project Structure

(as above)

\## Reproducibility

\- All random seeds set to 42

\- Package versions in \`requirements.txt\`

\- Data splits: Use official scikit-learn train/test

\## License & Citation

\...

**requirements.txt**

text

numpy==1.24.3

pandas==2.0.2

scikit-learn==1.3.0

scipy==1.11.1

matplotlib==3.7.2

seaborn==0.12.2

plotly==5.15.0

nltk==3.8.1

spacy==3.6.0

gensim==4.3.1

torch==2.0.1

torchvision==0.15.2

tensorflow==2.13.0

transformers==4.30.2

datasets==2.13.1

optuna==3.2.0

lime==0.2.0.1

shap==0.42.1

mlflow==2.4.1

jupyterlab==4.0.2

ipywidgets==8.0.6

**Experiment Tracking**

-   **MLflow**:

    -   Log: Parameters, metrics, artifacts (models, plots)

    -   UI: mlflow ui to browse runs

-   **Weights & Biases**:

    -   wandb.init(project=\'20newsgroups\', config=config)

    -   Automatic logging of metrics, system stats, model checkpoints

-   **TensorBoard** (for deep learning):

    -   SummaryWriter (PyTorch), TensorBoard callback (TensorFlow)

**Testing & Validation**

-   **Unit tests**: Test preprocessing functions, data loaders, model
    > forward passes

-   **Integration tests**: End-to-end pipeline (load → preprocess →
    > train → evaluate)

-   **Continuous Integration**: GitHub Actions to run tests on push

## **7.10 Final Report Structure**

**Executive Summary** (1-2 pages)

-   Objective, approach, key results

-   Best model: \[Name\], Test Macro-F1: \[X.XXX\], Inference: \[XX ms\]

-   Top insights: What worked, what didn\'t

**1. Introduction**

-   Background on 20 newsgroups dataset

-   Project goals and success criteria

-   Evaluation protocol (10-fold CV, test holdout)

**2. Exploratory Data Analysis**

-   Dataset statistics, class distribution

-   Text length analysis, vocabulary analysis

-   Metadata and leakage assessment

-   Category relationships, confusion-prone pairs

-   Key visualizations (10-15 figures)

**3. Data Preprocessing**

-   Text cleaning pipeline (headers removal, tokenization)

-   Normalization techniques (stop words, stemming/lemmatization)

-   Evaluation of preprocessing variants (table of macro-F1 results)

**4. Feature Engineering**

-   Sparse representations: BoW, TF-IDF (word, character, hybrid)

-   Embeddings: Word2Vec, GloVe, FastText

-   Feature selection and dimensionality reduction

-   Comparison table: Feature set vs CV macro-F1

**5. Baseline Modeling**

-   Classical models: NB, LogReg, SVM, RF, XGBoost

-   10-fold CV results (table, box plots)

-   Confusion matrices, per-class F1

-   Statistical significance tests

**6. Feature Engineering Refinement & Hyperparameter Tuning**

-   Iterative feature improvements guided by error analysis

-   Hyperparameter search strategies (Grid, Random, Bayesian)

-   Optimized model results

**7. Deep Learning Models**

-   Architecture descriptions: CNN, LSTM, Transformers

-   Training procedures and hyperparameters

-   Performance comparison: PyTorch vs TensorFlow models

-   Training curves, validation metrics

**8. Ensemble Methods**

-   Voting, stacking configurations

-   Ensemble performance vs individual models

-   Statistical significance of improvements

**9. Model Interpretability**

-   Global: Feature importance, attention aggregations

-   Local: LIME, SHAP examples

-   Insights: What drives predictions? Category-specific patterns

**10. Final Evaluation**

-   Test set results (macro-F1, accuracy, per-class metrics)

-   Confusion matrix analysis

-   Error taxonomy and failure modes

-   Comparison to literature benchmarks

**11. Model Selection & Deployment**

-   Selection criteria and decision matrix

-   Recommended model(s) for production

-   Inference pipeline, API design

-   Computational requirements (latency, memory)

**12. Conclusion**

-   Summary of findings

-   Key takeaways: Best practices for text classification

-   Limitations: What could be improved

-   Future work: Active learning, semi-supervised, domain adaptation

**13. References**

-   Papers, libraries, datasets cited

**Appendices**

-   **A**: Detailed hyperparameter grids

-   **B**: Full per-class metrics tables

-   **C**: Additional visualizations (embeddings, attention heatmaps)

-   **D**: Code snippets (key functions, model definitions)

## **7.11 Final Deliverables Checklist**

**Code & Models**

-   Complete source code in organized repository

-   Trained models and vectorizers (serialized, versioned)

-   Inference pipeline and API (tested, documented)

-   Unit and integration tests (passing)

-   Dockerfile and deployment instructions

**Data & Results**

-   Preprocessed datasets (saved, reproducible)

-   OOF predictions and test predictions (CSV/JSON)

-   All metrics logged (MLflow/W&B/CSV)

-   30-50 visualizations (EDA, results, interpretability)

**Documentation**

-   README with setup and usage instructions

-   requirements.txt / environment.yml (pinned versions)

-   Jupyter notebooks (executed, with markdown explanations)

-   Final report (PDF, 30-50 pages)

-   Presentation slides (optional, 15-20 slides)

**Reproducibility**

-   Random seeds set everywhere

-   Exact package versions documented

-   Data loading scripts (fetch_20newsgroups calls logged)

-   Experiment configs (YAML/JSON)

-   Instructions to reproduce results from scratch

```{=html}
<!-- -->
```
-   
