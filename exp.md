# Experiment Plan: Label Leakage in Author Profiling

## Original Experiment Table (from image)

| # | Experiment ID | Model | Main change / focus | Evaluation setup | Runs |
|---|---------------|-------|---------------------|------------------|------|
| 1 | Exp-1-baseline | distilbert-base-uncased | Baseline fine-tuning with default hyperparameters | Dataset-A-dev; metrics: accuracy, macro-F1 | 1 |
| 2 | Exp-2-hptune | distilbert-base-uncased | Hyperparameter tuning over LR {1e-5, 2e-5, 3e-5} and batch {16, 32} | Dataset-A-dev; best config selected by macro-F1 | 6 |
| 3 | Exp-3-larger-model | bert-base-uncased | Larger model, using best hyperparameters from Exp-2 | Dataset-A-dev; compared against Exp-1 and Exp-2 | 1 |
| 4 | Exp-4-augmentation | bert-base-uncased | Synonym-replacement data augmentation for 50% of training data | Dataset-A-dev; metrics: accuracy, macro-F1 | 1 |
| 5 | Exp-5-regularization | bert-base-uncased | Regularisation grid over dropout {0.1, 0.3} and weight decay {0, 0.01} | Dataset-A-dev; best config selected by macro-F1 | 4 |
| 6 | Exp-6-roberta-domain-adapt | roberta-base | Domain adaptation via additional pre-fine-tuning on unlabeled corpus | Dataset-A-dev; metrics: accuracy, macro-F1 | 1 |
| 7 | Exp-7-final-test | Best model from Exp-3–6 | Final evaluation of best configuration on in-domain and OOD test sets | Dataset-A-test and Dataset-B-test; metrics: accuracy, macro-F1 | 1 |

---

## Critical Analysis of Original Experiments

### Issues Identified

1. **Misalignment with Research Questions**: The original experiments focus on hyperparameter tuning and model architecture comparison, but the plan.tex RQs are about **label leakage** and **stylometric vs. transformer robustness**. There's a disconnect—experiments don't directly test H1/H2.

2. **Missing Stylometric Baseline**: The plan explicitly mentions a "character n-gram SVM" as a stylometric baseline, but it's absent from the experiment table.

3. **No Polluted vs. Cleaned Comparison**: The core hypothesis is about performance drop after removing demographic tokens. The experiments don't show this comparison.

4. **Vague Dataset References**: "Dataset-A" and "Dataset-B" are undefined. Need to specify: `gender.csv` (binary classification) or `birth_year.csv` (multi-class).

5. **Stylometric Approach Undefined**: Feedback requests clarification on what "stylometric" means operationally (character n-grams, function words, punctuation patterns, etc.).

6. **No Depollution Pipeline Evaluation**: How do we know the cleaning works? Need intrinsic evaluation of the cleaning step.

7. **Data Balance Not Addressed**: 
   - Gender: 53.3% male, 46.7% female (fairly balanced)
   - Birth year: 60 classes, highly imbalanced (1997: 2496 samples vs. many years <100)
   
8. **Qualitative Analysis Undefined**: Plan mentions qualitative inspection but doesn't specify methodology.

9. **No Heuristic Baselines**: Feedback requests both heuristic and learned baselines.

---

## Refined Research Questions

### RQ1 (Quantitative)
> To what extent do author-profiling models for **gender prediction on the Reddit Author Profiling Corpus** rely on explicit demographic mentions (e.g., "18F", "I'm a male", "as a woman") rather than genuine stylistic cues, as measured by the **relative drop in macro-F1** after removing label-leaking tokens?

### RQ2 (Comparative)
> Does **character n-gram SVM** (stylometric) exhibit a smaller relative performance degradation than **RoBERTa-base** (transformer) after demographic token removal, and what does this reveal about shortcut learning susceptibility?

### RQ3 (Qualitative)
> What types of predictions change after depollution, and do the changed predictions reveal systematic biases (e.g., topic-based gender stereotypes)?

---

## Refined Hypotheses

- **H1**: Removing explicit demographic cues will reduce macro-F1 by ≥10 percentage points for transformer models, indicating substantial shortcut exploitation.
- **H2**: Character n-gram SVM will show <50% of the relative performance drop compared to RoBERTa, demonstrating greater robustness to lexical shortcut removal.

---

## Clarification: Stylometric Approach

**Definition**: Stylometry refers to statistical analysis of linguistic style independent of semantic content. Our stylometric baseline uses:

1. **Character n-grams (3-5 grams)**: Capture morphological patterns, spelling habits, punctuation usage
2. **Function word frequencies**: Articles, pronouns, prepositions—content-independent
3. **Punctuation and capitalization patterns**: Sentence structure markers
4. **Sentence length statistics**: Mean, std, distribution

**Implementation**: LinearSVC with TF-IDF weighted character n-grams (n=3-5), max 50k features, L2 regularization.

**Rationale**: Character-level features are less susceptible to semantic shortcuts because they don't encode word-level demographic terms as strongly as contextual embeddings.

---

## Refined Experiment Plan

### Phase 0: Data Preparation and Depollution Validation

| ID | Experiment | Description | Metrics |
|----|------------|-------------|---------|
| **P0.1** | Leakage prevalence audit | Quantify % of samples containing explicit demographic tokens using regex patterns | Coverage %, token frequency distribution |
| **P0.2** | Depollution validation | Sample 200 cleaned texts; manual annotation of residual leakage | Precision of cleaning, false positive rate (meaningful content removed) |
| **P0.3** | Data split creation | Stratified 70/15/15 train/dev/test split by author_ID to prevent author leakage | Class distribution per split |

### Phase 1: Baseline Establishment

| ID | Experiment | Model | Data | Metrics |
|----|------------|-------|------|---------|
| **E1.1** | Majority baseline | Predict majority class | gender.csv (polluted) | Accuracy, macro-F1 |
| **E1.2** | Keyword heuristic | Rule-based: if contains "18F/22M/etc." → predict accordingly | gender.csv (polluted) | Precision, recall, coverage |
| **E1.3** | Stylometric SVM (polluted) | Char 3-5 gram SVM | gender.csv train (polluted) | Accuracy, macro-F1 on dev |
| **E1.4** | RoBERTa-base (polluted) | Fine-tuned RoBERTa | gender.csv train (polluted) | Accuracy, macro-F1 on dev |

### Phase 2: Core Hypothesis Testing (Polluted vs. Cleaned)

| ID | Experiment | Model | Train Data | Test Data | Comparison |
|----|------------|-------|------------|-----------|------------|
| **E2.1** | SVM on cleaned | Char n-gram SVM | gender_clean | gender_clean dev | vs. E1.3 |
| **E2.2** | RoBERTa on cleaned | RoBERTa-base | gender_clean | gender_clean dev | vs. E1.4 |
| **E2.3** | Cross-condition SVM | SVM trained on polluted | — | gender_clean dev | Generalization test |
| **E2.4** | Cross-condition RoBERTa | RoBERTa trained on polluted | — | gender_clean dev | Generalization test |

**Key Metrics**:
- Absolute macro-F1 drop: `F1_polluted - F1_cleaned`
- Relative drop: `(F1_polluted - F1_cleaned) / F1_polluted × 100%`
- Statistical significance: Paired bootstrap test (n=10000)

### Phase 3: Robustness and Ablations

| ID | Experiment | Description |
|----|------------|-------------|
| **E3.1** | Partial cleaning | Remove only compact tokens (18F, 22M) vs. full cleaning |
| **E3.2** | Masking vs. deletion | Compare [MASK] replacement vs. token deletion |
| **E3.3** | Training data size effect | Learning curves at 25%, 50%, 75%, 100% data |
| **E3.4** | DistilBERT comparison | Smaller transformer for efficiency/performance tradeoff |

### Phase 4: Final Evaluation

| ID | Experiment | Description |
|----|------------|-------------|
| **E4.1** | Test set evaluation | Best SVM and RoBERTa on held-out test (polluted and cleaned) |
| **E4.2** | Birth year transfer | Apply best models to birth_year.csv (multi-class, 60 classes) |
| **E4.3** | Confusion matrix analysis | Per-class performance, error patterns |

### Phase 5: Qualitative Analysis

| ID | Analysis | Method |
|----|----------|--------|
| **Q5.1** | Prediction flip analysis | Sample 100 instances where prediction changed post-cleaning; categorize by topic/content |
| **Q5.2** | Attention visualization | For RoBERTa: visualize attention on demographic vs. stylistic tokens |
| **Q5.3** | Feature importance | For SVM: top 50 most predictive n-grams before/after cleaning |
| **Q5.4** | Error taxonomy | Manually categorize 50 errors: stereotype-based, random, topic-based |

---

## Evaluation Metrics Justification

| Metric | Justification |
|--------|---------------|
| **Macro-F1** | Handles class imbalance; weights minority/majority equally |
| **Accuracy** | Interpretability baseline; less informative for imbalanced data |
| **Per-class precision/recall** | Identifies which demographic group is systematically misclassified |
| **Relative performance drop** | Core metric for H1/H2; normalizes for baseline performance |

**Note on Birth Year**: With 60 highly imbalanced classes, consider grouping into decades or using weighted macro-F1.

---

## Data Splitting Strategy

1. **Author-level split**: All posts from one author in same split (prevent identity leakage)
2. **Stratification**: Maintain class proportions across splits
3. **Reproducibility**: Fixed random seed (42)

---

## Baselines Summary

| Type | Baseline | Purpose |
|------|----------|---------|
| **Heuristic** | Majority class | Lower bound |
| **Heuristic** | Keyword matching | Tests if simple rules suffice |
| **Stylometric** | Char n-gram SVM | Content-independent features |
| **Neural** | DistilBERT | Efficient transformer |
| **Neural** | RoBERTa-base | Strong contextual model |

---

## Literature to Add (from ACL searches)

### Seminal Works
1. **Sap et al. (2014)** - Demographic prediction lexica from social media
2. **Gjurkovic & Snajder (2018)** - Reddit corpus for personality/demographic prediction (PEOPLES@NAACL)
3. **Sagawa et al. (2020)** - Group DRO for spurious correlations (ICLR)

### Recent Relevant Work
4. **Chen et al. (2024)** - "What Can Go Wrong in Authorship Profiling: Cross-Domain Analysis of Gender and Age Prediction" (GeBNLP@ACL 2024) - Shows models learn dataset-specific features and topical biases
5. **Zhou et al. (2024)** - "Navigating the Shortcut Maze: Comprehensive Analysis of Shortcut Learning in Text Classification" (Findings EMNLP 2024) - Categorizes shortcuts into occurrence, style, concept
6. **PAN Shared Tasks (2020-2023)** - Author profiling benchmarks and methodologies

### Narrative
The literature establishes that (1) social media text contains explicit demographic signals (Sap), (2) Reddit specifically enables large-scale profiling (Gjurkovic), but (3) models often exploit spurious correlations (Sagawa, Zhou) and (4) cross-domain evaluation reveals dataset-specific artifacts (Chen). Our work directly quantifies this leakage and compares stylometric vs. neural susceptibility.

---

## Timeline Suggestion

| Week | Activities |
|------|------------|
| 1 | P0: Data prep, cleaning pipeline, validation |
| 2 | E1: Baselines (majority, keyword, SVM, RoBERTa polluted) |
| 3 | E2: Core experiments (cleaned conditions) |
| 4 | E3: Ablations; E4: Final evaluation |
| 5 | Q5: Qualitative analysis; writing |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Incomplete leakage removal | Manual validation (P0.2); iterative regex refinement |
| Cleaning removes meaningful style | Compare text length distributions; qualitative review |
| Compute constraints for RoBERTa | Use DistilBERT as fallback; reduce batch size |
| Birth year too imbalanced | Focus on gender; treat birth year as secondary |

---

## Suggested Follow-up Experiments (if time permits)

1. **Adversarial debiasing**: Train with gradient reversal on demographic classifier
2. **Counterfactual augmentation**: Swap gender terms and measure prediction stability  
3. **Cross-dataset transfer**: Train on Reddit, test on PAN Twitter data
4. **Fine-grained leakage**: Distinguish self-identification ("I'm female") from third-party mentions
