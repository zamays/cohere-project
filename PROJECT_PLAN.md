# Cohere Pets Prior Authorization ML Project - Phased Plan

## Project Overview

**Business Goal**: Develop a machine learning model to auto-approve veterinary prior authorizations at submission time, reducing manual review workload and speeding up the authorization process.

**Data Available**:
- Prior Authorization Data: Historical auth requests with outcomes (2023-2024)
- Claims Data: Historical claims from insurance companies (2021-2023)
- Link: Pet ID connects the datasets

**Timeline**: 48 hours for completion
**Deliverables**: Jupyter notebook + 20-30 min presentation for moderately technical stakeholders

---

## Phase 1: Exploratory Data Analysis (EDA)

### Objectives
Understand the data structure, quality, and relationships to inform modeling decisions.

### Tasks

#### 1.1 Load and Inspect Datasets
- [ ] Load Prior Authorization data
- [ ] Load Claims data
- [ ] Examine shape, columns, and data types
- [ ] Preview first/last rows of each dataset

#### 1.2 Data Quality Assessment
- [ ] Check for missing values in each column
- [ ] Identify duplicates
- [ ] Analyze data distributions
- [ ] Check date ranges and temporal coverage

#### 1.3 Target Variable Analysis
- [ ] Examine `authstatus` distribution (Approved/Denied/Pending)
- [ ] Assess class imbalance
- [ ] Determine if Pending should be included or excluded
- [ ] Analyze `auto_approved` field relationship to `authstatus`

#### 1.4 Feature Analysis
**Prior Auth Dataset:**
- [ ] `service`: Unique values, distribution
- [ ] `provider_id`: Cardinality, distribution
- [ ] `unit`: Distribution, outliers
- [ ] `submission_date`: Temporal patterns
- [ ] `clinical_reviewer`: Missing data pattern, relevance for prediction

**Claims Dataset:**
- [ ] `Claim Type`: Distribution
- [ ] `Amount Paid`: Statistical summary, outliers
- [ ] `Claim Date`: Temporal patterns

#### 1.5 Dataset Relationship Analysis
- [ ] How many pets appear in both datasets?
- [ ] What % of prior auths have corresponding claim history?
- [ ] Temporal relationship: Are claims before or after auths?

#### 1.6 Key Questions to Answer
- [ ] What features are available at prediction time (auth submission)?
- [ ] Is there data leakage risk (e.g., `clinical_reviewer` only assigned after review)?
- [ ] What is the appropriate train/test split strategy (temporal vs random)?

### Deliverables
- Summary statistics tables
- Distribution visualizations
- Missing data heatmap
- Target variable distribution chart
- Data quality assessment report

---

## Phase 2: Data Cleaning & Feature Engineering

### Objectives
Prepare clean dataset with engineered features that maximize predictive power.

### Tasks

#### 2.1 Data Cleaning
- [ ] Handle missing values (strategy per column)
- [ ] Standardize Pet ID format across datasets
- [ ] Remove duplicates if any
- [ ] Validate date formats
- [ ] Handle outliers in `Amount Paid` and `unit`
- [ ] Decide on `clinical_reviewer` (likely exclude due to data leakage)

#### 2.2 Feature Engineering from Claims History

**CRITICAL**: Only use claims that occurred BEFORE the auth submission_date!

**Features to create per pet:**
- [ ] `claim_count_total`: Total number of historical claims
- [ ] `claim_count_last_30d`: Claims in last 30 days before auth
- [ ] `claim_count_last_90d`: Claims in last 90 days before auth
- [ ] `total_amount_paid`: Cumulative amount paid for this pet
- [ ] `avg_amount_paid`: Average amount per claim
- [ ] `max_amount_paid`: Highest single claim amount
- [ ] `days_since_last_claim`: Recency of last claim
- [ ] `unique_claim_types`: Number of different procedure types
- [ ] `has_claim_history`: Binary flag (0/1)

**Service-specific features:**
- [ ] `prior_claims_same_service`: Count of same procedure type
- [ ] `avg_paid_same_service`: Average paid for same procedure

**Provider features:**
- [ ] `provider_approval_rate`: Historical approval rate for this provider
- [ ] `provider_auth_count`: Number of auths from this provider

#### 2.3 Temporal Features
- [ ] `day_of_week`: Day of week submitted (0-6)
- [ ] `month`: Month of submission (1-12)
- [ ] `quarter`: Quarter of submission (1-4)
- [ ] `is_weekend`: Boolean flag
- [ ] `days_since_year_start`: Days since Jan 1

#### 2.4 Categorical Encoding
- [ ] One-hot encode `service` (if cardinality is reasonable)
- [ ] Consider target encoding for high-cardinality features
- [ ] Encode `authstatus` as target variable (binary or multiclass)

#### 2.5 Feature Scaling
- [ ] Identify which features need scaling
- [ ] Apply StandardScaler or RobustScaler to numeric features
- [ ] Document scaling decisions

#### 2.6 Train/Test Split
- [ ] **Use temporal split** (not random) to avoid data leakage
- [ ] Recommendation: 80% train / 20% test based on submission_date
- [ ] Ensure no data leakage across split

### Deliverables
- Cleaned dataset
- Engineered features dataframe
- Feature engineering code (reusable)
- Train/test split datasets
- Documentation of all transformations

---

## Phase 3: Model Development

### Objectives
Build and compare multiple ML models to auto-approve prior authorizations, with special focus on reducing False Negatives while maintaining acceptable precision.

### Tasks

#### 3.1 Baseline Model
- [ ] Create simple rule-based baseline (e.g., majority class)
- [ ] Logistic Regression as statistical baseline
- [ ] Document baseline performance metrics

#### 3.2 Advanced Models
- [ ] Random Forest Classifier
- [ ] XGBoost / LightGBM
- [ ] Consider ensemble methods

#### 3.3 Handling Class Imbalance (Critical for Reducing False Negatives)

**Problem**: Prior authorization datasets typically have imbalanced classes (more approvals than denials), leading to models that miss many actual approvals (high False Negative rate).

**Strategies to implement:**

**3.3.1 Data-Level Approaches:**
- [ ] **SMOTE (Synthetic Minority Over-sampling Technique)**: Generate synthetic samples for minority class
- [ ] **ADASYN (Adaptive Synthetic Sampling)**: Focus on harder-to-learn minority samples
- [ ] **Random Under-sampling**: Reduce majority class samples (use with caution)
- [ ] **Combination Sampling**: Mix over-sampling and under-sampling (SMOTETomek, SMOTEENN)
- [ ] **Time-aware sampling**: Ensure temporal integrity when resampling

**3.3.2 Algorithm-Level Approaches:**
- [ ] **Class Weights**: Set `class_weight='balanced'` or custom weights inversely proportional to class frequencies
- [ ] **Focal Loss**: Modify loss function to focus on hard-to-classify examples (especially for neural networks)
- [ ] **Cost-Sensitive Learning**: Assign different misclassification costs (False Negative cost > False Positive cost)
- [ ] **Sample Weights**: Weight training samples based on class membership and importance

**3.3.3 Ensemble Approaches:**
- [ ] **Balanced Random Forest**: Each tree trained on balanced bootstrap sample
- [ ] **EasyEnsemble**: Create multiple balanced subsets and train separate models
- [ ] **BalancedBagging**: Bagging with balanced bootstrap samples
- [ ] **Balanced Boosting**: Boosting algorithms with balanced sampling at each iteration

#### 3.4 Advanced Feature Engineering for Improved Recall

**Interaction Features:**
- [ ] Service × Provider interactions (certain providers may have patterns for services)
- [ ] Service × Claim History interactions (history predictiveness varies by service)
- [ ] Temporal × Service interactions (seasonal patterns for certain procedures)

**Aggregated Claim Features:**
- [ ] Rolling statistics (7-day, 30-day, 90-day windows): mean, median, std of claim amounts
- [ ] Claim velocity: Rate of claims over time
- [ ] Claim patterns: Regularity/irregularity of claim timing
- [ ] Service diversity score: Entropy or count of unique services

**Risk Indicators:**
- [ ] High-risk service flags (based on historical denial rates)
- [ ] Provider reliability score (consistency of approval rates)
- [ ] Unusual claim amount flags (outliers in amount_paid)
- [ ] Gap in claim history flags (long periods without claims)

**Polynomial Features:**
- [ ] Quadratic interactions between key numeric features
- [ ] Only for most important features to avoid dimensionality curse

#### 3.5 Model Training

- [ ] Define evaluation metrics (Precision, Recall, F1, ROC-AUC)
- [ ] **Custom scoring function**: Weighted combination of Precision and Recall based on business priorities
- [ ] Implement cross-validation (time-series aware with StratifiedKFold for each temporal split)
- [ ] Hyperparameter tuning (GridSearch or RandomSearch)
  - [ ] Include class_weight, scale_pos_weight (XGBoost), and threshold as hyperparameters
  - [ ] Optimize for recall-focused metrics (F-beta score with beta=2 prioritizes recall)
- [ ] Track experiments and results

#### 3.6 Threshold Optimization for Balanced Performance

**Problem**: Default 0.5 threshold may not be optimal for business objectives.

**Strategies:**
- [ ] **Precision-Recall Curve Analysis**: Find threshold that balances precision/recall
- [ ] **Cost-Based Threshold**: Assign costs to FP and FN, find threshold that minimizes total cost
- [ ] **F-beta Optimization**: Use F2 score (emphasizes recall) or F0.5 (emphasizes precision)
- [ ] **Business Constraint Threshold**: E.g., "maintain 70% precision while maximizing recall"
- [ ] **Multi-threshold Strategy**: 
  - High confidence (>0.8): Auto-approve
  - Medium confidence (0.5-0.8): Enhanced review (prioritize)
  - Low confidence (<0.5): Standard manual review

#### 3.7 Model Calibration

**Purpose**: Ensure predicted probabilities reflect true likelihoods (important for threshold-based decisions).

**Methods:**
- [ ] **Platt Scaling**: Fit logistic regression on model outputs
- [ ] **Isotonic Regression**: Non-parametric calibration
- [ ] **Beta Calibration**: Extension of Platt scaling for more flexibility
- [ ] Evaluate calibration with calibration curves and Brier score

#### 3.8 Ensemble Methods

**Stacking:**
- [ ] Use predictions from multiple models as meta-features
- [ ] Train meta-learner (e.g., Logistic Regression) on diverse base models
- [ ] Include calibrated probability predictions

**Weighted Voting:**
- [ ] Weight models by their recall performance (since reducing FN is priority)
- [ ] Optimize weights using validation set

**Blending:**
- [ ] Hold-out set for training meta-learner (simpler than stacking)

#### 3.9 Model Evaluation

**Metrics to calculate:**
- [ ] Precision (avoid false approvals)
- [ ] Recall (capture as many approvals as possible) - **PRIMARY METRIC**
- [ ] F1-Score (balance precision/recall)
- [ ] F2-Score (emphasizes recall more than precision)
- [ ] ROC-AUC
- [ ] Precision-Recall AUC (better for imbalanced datasets)
- [ ] Confusion matrix with detailed FN analysis
- [ ] Classification report per class
- [ ] False Negative Rate by service type (identify where model struggles)

**Business-specific metrics:**
- [ ] % of auths that can be auto-approved at high confidence
- [ ] False approval rate (business risk)
- [ ] **False Negative rate** (missed opportunity cost)
- [ ] Cost-benefit analysis with explicit FN costs
- [ ] Workload reduction vs. risk trade-off analysis

#### 3.10 Feature Importance Analysis
- [ ] Extract feature importances from best model
- [ ] Visualize top 20 most important features
- [ ] Validate that features make business sense
- [ ] Check for unexpected dependencies
- [ ] Analyze feature importance for FN cases specifically

#### 3.11 Model Interpretation
- [ ] SHAP values for explainability
- [ ] Analyze edge cases and failure modes
- [ ] **Deep dive into False Negatives**: What characteristics do missed approvals have?
- [ ] Document model limitations
- [ ] Create interpretable rules for borderline cases

### Deliverables
- Trained models (pickle files) with optimal hyperparameters and class balancing
- Performance comparison table with emphasis on Recall and F2-Score
- ROC curves, Precision-Recall curves, and confusion matrices
- Calibration curves showing probability calibration quality
- Feature importance visualizations
- Model evaluation report with detailed False Negative analysis
- Threshold analysis charts showing Precision/Recall trade-offs
- Business impact analysis with different recall targets

---

## Phase 4: Business Recommendation

### Objectives
Provide actionable recommendation on deploying the ML model with appropriate balance between False Negatives (missed approvals) and False Positives (wrong approvals).

### Tasks

#### 4.1 Performance Analysis
- [ ] Compare ML model vs current rule-based system
- [ ] Quantify improvement in auto-approval rate
- [ ] Analyze error types and business impact
- [ ] **Detailed False Negative Analysis:**
  - [ ] Cost of manual review for missed approvals (labor cost × FN count)
  - [ ] Delay cost for customers (slower approval time)
  - [ ] Customer satisfaction impact
  - [ ] Revenue impact if delays lead to cancellations

#### 4.2 Risk Assessment
- [ ] Calculate false positive rate (wrongly auto-approved)
- [ ] Calculate false negative rate (missed approvals) **- KEY METRIC**
- [ ] Estimate financial impact of errors:
  - **False Positives**: Direct cost of incorrect approvals (fraud, claims, reputation)
  - **False Negatives**: Opportunity cost (manual review time, delays, customer dissatisfaction)
- [ ] Assess reputational risk for both FP and FN
- [ ] Create risk matrix: FP cost vs. FN cost

#### 4.3 Cost-Benefit Analysis with FN Considerations

**False Negative Impact:**
- [ ] Manual review cost per authorization: $X
- [ ] Average delay caused by manual review: Y hours
- [ ] Customer churn risk from delays: Z%
- [ ] Total FN cost = (FN_count × review_cost) + (FN_count × delay_cost)

**False Positive Impact:**
- [ ] Average cost of incorrect approval: $W
- [ ] Reputation damage cost: $R
- [ ] Total FP cost = (FP_count × approval_cost) + reputation_cost

**Optimal Operating Point:**
- [ ] Find threshold where: (FP_cost × FP_rate) + (FN_cost × FN_rate) is minimized
- [ ] Consider multiple thresholds for different scenarios

#### 4.4 Business Impact Projection
- [ ] % of auths that can be auto-approved
- [ ] Reduction in manual review workload
- [ ] Time savings for approval process
- [ ] **Improvement in recall** (reduction in False Negatives)
- [ ] Customer experience improvement (faster approvals)
- [ ] Revenue implications

#### 4.5 Multi-Tier Approval Strategy

Instead of binary auto-approve/manual-review, consider three tiers:

**Tier 1: High Confidence Auto-Approve (e.g., probability > 0.85)**
- [ ] Define precision target (e.g., 95%+)
- [ ] Estimate recall at this threshold
- [ ] Calculate volume and business impact

**Tier 2: Medium Confidence - Accelerated Review (e.g., probability 0.60-0.85)**
- [ ] Flag for expedited human review (prioritize in queue)
- [ ] Provide ML confidence score to reviewer
- [ ] Suggest approval with human verification
- [ ] Estimate recall improvement from including this tier

**Tier 3: Low Confidence - Standard Review (e.g., probability < 0.60)**
- [ ] Route to standard manual review process
- [ ] No ML recommendation shown (avoid bias)

**Benefits of Multi-Tier:**
- [ ] Captures more approvals (reduces FN) while maintaining safety
- [ ] Provides flexibility in workload management
- [ ] Better customer experience for medium-confidence cases

#### 4.6 Implementation Considerations
- [ ] Model deployment requirements
- [ ] Monitoring and maintenance needs
- [ ] Fallback to human review strategy
- [ ] Confidence threshold recommendations (with rationale)
- [ ] **A/B testing plan**: Compare FN rates with and without ML
- [ ] Feedback loop: Capture human decisions on model predictions for continuous improvement

#### 4.7 Final Recommendation
- [ ] Deploy or not deploy decision
- [ ] Justification with data (including FN analysis)
- [ ] **Recommended threshold(s)** with clear trade-off explanation
- [ ] Phased rollout strategy if applicable
- [ ] Success metrics for monitoring (include FN rate targets)
- [ ] Plan for continuous model improvement to reduce FN over time

### Deliverables
- Business recommendation document with detailed FN/FP trade-off analysis
- ROI analysis including costs of False Negatives
- Risk mitigation strategy addressing both FP and FN
- Multi-tier approval strategy recommendation
- Implementation roadmap with A/B testing plan
- Monitoring dashboard requirements (including FN rate tracking)

---

## Phase 4.5: Comprehensive Strategies for Reducing False Negatives

### Objectives
Provide concrete, actionable strategies to reduce False Negative rate while maintaining acceptable precision.

### Understanding the False Negative Problem

**Current State Analysis:**
- High FN rate (e.g., 95%+) means model is too conservative
- Model optimizes for precision at extreme cost to recall
- Business impact: Most approvals go to manual review, reducing automation benefits
- Root causes: Class imbalance, overly high threshold, insufficient features, model bias

### Strategy 1: Class Imbalance Correction

#### 1.1 Implement SMOTE (Synthetic Minority Over-sampling)
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

**Benefits:**
- Creates synthetic samples for minority class
- Helps model learn minority class patterns better
- Reduces bias toward majority class

**Considerations:**
- Apply only to training data (not test)
- May create noise if not enough features
- Use with temporal awareness (don't leak future data)

#### 1.2 Adjust Class Weights
```python
# For XGBoost
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
# More aggressive: multiply by factor (e.g., 1.5x or 2x)
scale_pos_weight = scale_pos_weight * 1.5

# For sklearn models
class_weight = {0: 1, 1: 2.0}  # Weight minority class higher
```

**Benefits:**
- Simple to implement
- Forces model to pay more attention to minority class
- Can be tuned as hyperparameter

### Strategy 2: Optimize Decision Threshold

#### 2.1 Use Precision-Recall Curve
```python
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Find threshold for target recall (e.g., 50% recall)
target_recall = 0.50
idx = np.argmin(np.abs(recall - target_recall))
optimal_threshold = thresholds[idx]
```

#### 2.2 Cost-Based Threshold Optimization
```python
# Define costs
cost_fn = 10  # Cost of False Negative (missed approval - manual review cost)
cost_fp = 50  # Cost of False Positive (wrong approval - fraud/reputation cost)

# Find threshold that minimizes total cost
min_cost = float('inf')
best_threshold = 0.5
for threshold in np.arange(0.1, 0.95, 0.05):
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    if total_cost < min_cost:
        min_cost = total_cost
        best_threshold = threshold
```

#### 2.3 F-beta Score Optimization (Emphasize Recall)
```python
from sklearn.metrics import fbeta_score

# F2 score weights recall 2x more than precision
# Find threshold that maximizes F2
best_f2 = 0
best_threshold = 0.5
for threshold in np.arange(0.1, 0.95, 0.05):
    y_pred = (y_pred_proba >= threshold).astype(int)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    if f2 > best_f2:
        best_f2 = f2
        best_threshold = threshold
```

### Strategy 3: Advanced Feature Engineering

#### 3.1 Create Recall-Focused Features

**Features that help identify approvals:**
- Historical approval patterns for similar cases
- Provider success rate with specific service types
- Pet claim history consistency (regular claims often indicate established care)
- Service-specific features (some services are almost always approved)

```python
# Example: Provider-service approval rate
provider_service_approval = train_data.groupby(['provider_id', 'service'])['target'].mean()
data['provider_service_approval_rate'] = data.apply(
    lambda x: provider_service_approval.get((x['provider_id'], x['service']), 0.5), axis=1
)
```

#### 3.2 Interaction Features
```python
# Create polynomial features for key predictors
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X[['claim_count_total', 'avg_amount_paid', 'unit']])
```

### Strategy 4: Model Architecture Improvements

#### 4.1 Use Recall-Focused Evaluation Metric in Training
```python
from sklearn.metrics import make_scorer, fbeta_score

# Custom scorer that prioritizes recall
f2_scorer = make_scorer(fbeta_score, beta=2)

# Use in GridSearchCV
grid_search = GridSearchCV(
    model, 
    param_grid, 
    scoring=f2_scorer,  # Optimize for F2 instead of accuracy
    cv=tscv
)
```

#### 4.2 Ensemble with Recall-Focused Models
```python
# Train separate models optimized for different objectives
model_precision = XGBClassifier(scale_pos_weight=scale_pos_weight, max_depth=4)
model_recall = XGBClassifier(scale_pos_weight=scale_pos_weight*2, max_depth=8)

# Combine predictions with weighted average
y_pred_proba = 0.4 * model_precision.predict_proba(X_test)[:, 1] + \
               0.6 * model_recall.predict_proba(X_test)[:, 1]
```

#### 4.3 Calibrate Probabilities
```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrate to get better probability estimates
calibrated_model = CalibratedClassifierCV(xgb_model, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)
y_pred_proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
```

### Strategy 5: Multi-Threshold Approach

Instead of single threshold, use multiple tiers:

```python
# Define three confidence tiers
high_confidence_threshold = 0.75  # Auto-approve (maintain ~90% precision)
medium_confidence_threshold = 0.45  # Accelerated review (capture more recalls)

# Classify into tiers
def classify_with_tiers(proba):
    if proba >= high_confidence_threshold:
        return 'auto_approve'
    elif proba >= medium_confidence_threshold:
        return 'expedited_review'
    else:
        return 'standard_review'

# Apply
predictions = np.array([classify_with_tiers(p) for p in y_pred_proba])

# Metrics for each tier
auto_approve_mask = predictions == 'auto_approve'
print(f"Auto-approve: {auto_approve_mask.sum()} cases")
print(f"Precision: {precision_score(y_test[auto_approve_mask], y_pred[auto_approve_mask])}")
print(f"Recall: {recall_score(y_test, auto_approve_mask.astype(int))}")
```

### Strategy 6: Active Learning Loop

#### 6.1 Identify High-Uncertainty Cases
```python
# Find cases near decision boundary
uncertainty_mask = (y_pred_proba >= 0.4) & (y_pred_proba <= 0.6)
uncertain_cases = X_test[uncertainty_mask]

# Prioritize these for human review and feedback
# Use feedback to retrain model
```

#### 6.2 Continuous Model Updates
- Collect human decisions on model predictions
- Retrain model monthly/quarterly with new data
- Focus on cases where model had low confidence or made mistakes

### Strategy 7: Separate Models by Service Type

```python
# Different services may have different approval patterns
services = train_data['service'].unique()

models = {}
for service in services:
    service_mask = train_data['service'] == service
    X_service = X_train[service_mask]
    y_service = y_train[service_mask]
    
    # Train service-specific model
    model = XGBClassifier(scale_pos_weight=service_specific_weight)
    model.fit(X_service, y_service)
    models[service] = model

# Predict using appropriate model
def predict_by_service(X_test, service):
    return models[service].predict_proba(X_test)[:, 1]
```

### Expected Impact

**Target Metrics After Implementing These Strategies:**

| Metric | Current | Target | Impact |
|--------|---------|--------|--------|
| False Negative Rate | 95.76% | 30-50% | Reduce by 45-65 percentage points |
| Recall | 4.24% | 50-70% | Increase by 10-16x |
| Precision | 75% | 70-80% | Maintain similar levels |
| Auto-Approval Rate | 4.16% | 30-50% | Increase by 7-12x |
| F1-Score | 0.08 | 0.60-0.75 | Significant improvement |

**Business Impact:**
- 30-50% of authorizations auto-approved (vs. 4% currently)
- Significant reduction in manual review workload
- Faster approval times for customers
- Better model utilization and ROI

### Implementation Checklist

**Phase 1: Quick Wins (Immediate)**
- [ ] Adjust class weights (multiply scale_pos_weight by 1.5-2x)
- [ ] Lower decision threshold from 0.9 to 0.5-0.7
- [ ] Use F2-score for model evaluation

**Phase 2: Data Improvements (1 week)**
- [ ] Implement SMOTE or ADASYN
- [ ] Add interaction features (provider × service)
- [ ] Create provider-service approval rate features

**Phase 3: Model Refinement (1-2 weeks)**
- [ ] Retrain models with F2-scorer in GridSearchCV
- [ ] Implement probability calibration
- [ ] Test ensemble methods
- [ ] Optimize threshold using cost-based approach

**Phase 4: Production Strategy (2-3 weeks)**
- [ ] Implement multi-threshold approach (3 tiers)
- [ ] Set up A/B testing framework
- [ ] Create monitoring dashboard for FN rate
- [ ] Establish feedback loop for continuous improvement

### Validation

**Before deployment, verify:**
- [ ] FN rate reduced to target level (30-50%)
- [ ] Precision maintained at acceptable level (70%+)
- [ ] Model performance consistent across service types
- [ ] Business cost-benefit analysis favors deployment
- [ ] Stakeholders approve trade-off between precision and recall

---

## Phase 5: LLM Chatbot Evaluation Design

### Objectives
Design an evaluation framework for the LLM-based clinical notes summarization chatbot.

### Tasks

#### 5.1 Define Evaluation Dimensions
- [ ] **Accuracy**: Correctness against ground truth
- [ ] **Completeness**: Coverage of relevant clinical facts
- [ ] **Consistency**: Test-retest reliability
- [ ] **Hallucination Detection**: Fabricated information
- [ ] **Usefulness**: Time savings and decision support quality

#### 5.2 Evaluation Methods

**Accuracy:**
- [ ] Create gold-standard summaries (expert-reviewed)
- [ ] Compare LLM output to gold standard (F1, ROUGE, BLEU)
- [ ] Manual review by domain experts

**Completeness:**
- [ ] Checklist of required clinical elements
- [ ] Recall-focused metrics
- [ ] Missing information analysis

**Consistency:**
- [ ] Same input multiple times (temperature = 0)
- [ ] Compare outputs for variability
- [ ] Inter-rater reliability metrics

**Hallucination:**
- [ ] Fact verification against source notes
- [ ] Expert review for fabricated claims
- [ ] Automated fact-checking pipeline

**Usefulness:**
- [ ] A/B testing with human reviewers
- [ ] Time-to-decision metrics
- [ ] User satisfaction surveys
- [ ] Decision accuracy with/without chatbot

#### 5.3 Evaluation Dataset
- [ ] Sample size determination
- [ ] Stratified sampling (by service type, complexity)
- [ ] Diverse edge cases

#### 5.4 Strengths and Limitations
- [ ] Document what evaluation captures well
- [ ] Document what evaluation may miss
- [ ] Identify blindspots

#### 5.5 Recommendation
- [ ] Explain why this approach is appropriate
- [ ] Cost-benefit of evaluation
- [ ] Continuous monitoring strategy

### Deliverables
- LLM evaluation framework document
- Evaluation metrics specification
- Strengths/limitations analysis
- Implementation recommendation

---

## Phase 6: Presentation Preparation

### Objectives
Create compelling 20-30 min presentation for moderately technical stakeholders.

### Tasks

#### 6.1 Slide Deck Structure
1. **Introduction** (2 min)
   - [ ] Problem statement
   - [ ] Business context
   - [ ] Project objectives

2. **Data Overview** (3 min)
   - [ ] Datasets description
   - [ ] Data quality insights
   - [ ] Key statistics

3. **Approach** (4 min)
   - [ ] Feature engineering strategy
   - [ ] Model selection rationale
   - [ ] Evaluation methodology

4. **Results** (6 min)
   - [ ] Model performance metrics
   - [ ] Feature importance
   - [ ] Comparison to baseline

5. **Business Recommendation** (5 min)
   - [ ] Deploy or not decision
   - [ ] Expected impact
   - [ ] Risk mitigation

6. **LLM Evaluation Design** (5 min)
   - [ ] Proposed framework
   - [ ] Strengths and limitations
   - [ ] Why this approach

7. **Next Steps** (2 min)
   - [ ] Implementation plan
   - [ ] Success metrics
   - [ ] Q&A

#### 6.2 Visualization Preparation
- [ ] Clean, professional charts
- [ ] Consistent color scheme
- [ ] Minimal text, maximum clarity
- [ ] Annotate key insights

#### 6.3 Storytelling
- [ ] Clear narrative arc
- [ ] Connect technical to business value
- [ ] Anticipate questions
- [ ] Practice delivery

### Deliverables
- 20-30 min slide deck (PowerPoint/Google Slides)
- Speaker notes
- Appendix with additional details

---

## Success Criteria

### Technical Excellence
- [ ] Clean, well-documented code
- [ ] Reproducible results
- [ ] Proper validation methodology
- [ ] No data leakage

### Business Acumen
- [ ] Clear ROI analysis
- [ ] Risk-aware recommendations
- [ ] Practical implementation plan
- [ ] Stakeholder-appropriate communication

### Communication
- [ ] Clear presentation narrative
- [ ] Effective visualizations
- [ ] Defended technical decisions
- [ ] Anticipate and answer questions

---

## Timeline (48 hours)

**Hours 0-8**: Phase 1 - EDA
**Hours 8-16**: Phase 2 - Data Cleaning & Feature Engineering
**Hours 16-28**: Phase 3 - Model Development
**Hours 28-34**: Phase 4 - Business Recommendation
**Hours 34-40**: Phase 5 - LLM Evaluation Design
**Hours 40-48**: Phase 6 - Presentation Preparation

---

## Key Principles

1. **Avoid Data Leakage**: Only use information available at auth submission time
2. **Temporal Awareness**: Use time-based train/test split
3. **Business Focus**: Optimize for business value, not just accuracy
4. **Explainability**: Ensure model decisions can be explained to stakeholders
5. **Risk Management**: Quantify and communicate prediction errors
6. **Practicality**: Design for real-world deployment

---

## Notes

- This plan is a living document and may be adjusted based on findings during EDA
- Priority is delivering a defensible, business-ready solution over perfect technical sophistication
- Focus on clear communication of trade-offs and decisions made
