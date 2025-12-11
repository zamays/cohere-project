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
Build and compare multiple ML models to auto-approve prior authorizations.

### Tasks

#### 3.1 Baseline Model
- [ ] Create simple rule-based baseline (e.g., majority class)
- [ ] Logistic Regression as statistical baseline
- [ ] Document baseline performance metrics

#### 3.2 Advanced Models
- [ ] Random Forest Classifier
- [ ] XGBoost / LightGBM
- [ ] Consider ensemble methods

#### 3.3 Model Training
- [ ] Define evaluation metrics (Precision, Recall, F1, ROC-AUC)
- [ ] Implement cross-validation (time-series aware)
- [ ] Hyperparameter tuning (GridSearch or RandomSearch)
- [ ] Track experiments and results

#### 3.4 Model Evaluation

**Metrics to calculate:**
- [ ] Precision (avoid false approvals)
- [ ] Recall (capture as many approvals as possible)
- [ ] F1-Score (balance precision/recall)
- [ ] ROC-AUC
- [ ] Confusion matrix
- [ ] Classification report per class

**Business-specific metrics:**
- [ ] % of auths that can be auto-approved at high confidence
- [ ] False approval rate (business risk)
- [ ] Cost-benefit analysis

#### 3.5 Feature Importance Analysis
- [ ] Extract feature importances from best model
- [ ] Visualize top 20 most important features
- [ ] Validate that features make business sense
- [ ] Check for unexpected dependencies

#### 3.6 Model Interpretation
- [ ] SHAP values for explainability
- [ ] Analyze edge cases and failure modes
- [ ] Document model limitations

### Deliverables
- Trained models (pickle files)
- Performance comparison table
- ROC curves and confusion matrices
- Feature importance visualizations
- Model evaluation report

---

## Phase 4: Business Recommendation

### Objectives
Provide actionable recommendation on deploying the ML model.

### Tasks

#### 4.1 Performance Analysis
- [ ] Compare ML model vs current rule-based system
- [ ] Quantify improvement in auto-approval rate
- [ ] Analyze error types and business impact

#### 4.2 Risk Assessment
- [ ] Calculate false positive rate (wrongly auto-approved)
- [ ] Calculate false negative rate (missed approvals)
- [ ] Estimate financial impact of errors
- [ ] Assess reputational risk

#### 4.3 Business Impact Projection
- [ ] % of auths that can be auto-approved
- [ ] Reduction in manual review workload
- [ ] Time savings for approval process
- [ ] Revenue implications

#### 4.4 Implementation Considerations
- [ ] Model deployment requirements
- [ ] Monitoring and maintenance needs
- [ ] Fallback to human review strategy
- [ ] Confidence threshold recommendations

#### 4.5 Final Recommendation
- [ ] Deploy or not deploy decision
- [ ] Justification with data
- [ ] Phased rollout strategy if applicable
- [ ] Success metrics for monitoring

### Deliverables
- Business recommendation document
- ROI analysis
- Risk mitigation strategy
- Implementation roadmap

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
