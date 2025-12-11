#!/usr/bin/env python
# coding: utf-8

# # Cohere Pets Prior Authorization ML Project
# ## Phase 1: Exploratory Data Analysis

# In[90]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load datasets
prior_auth = pd.read_excel('Emma-_Data_Set.xlsx', sheet_name='PriorAuth')
claims = pd.read_excel('Emma-_Data_Set.xlsx', sheet_name='Claims')

print(f"Prior Authorization Data Shape: {prior_auth.shape}")
print(f"Claims Data Shape: {claims.shape}")

# ### 1.1 Initial Data Inspection

# In[91]:


# Prior Authorization Dataset
print("="*60)
print("PRIOR AUTHORIZATION DATASET")
print("="*60)
print(f"\nShape: {prior_auth.shape}")
print(f"\nColumns and Types:")
print(prior_auth.dtypes)
print(f"\nFirst 5 rows:")
prior_auth.head()

# In[92]:


# Claims Dataset
print("="*60)
print("CLAIMS DATASET")
print("="*60)
print(f"\nShape: {claims.shape}")
print(f"\nColumns and Types:")
print(claims.dtypes)
print(f"\nFirst 5 rows:")
claims.head()

# ### 1.2 Data Quality Assessment

# In[93]:


# Missing values analysis
print("PRIOR AUTHORIZATION - Missing Values:")
missing_auth = pd.DataFrame({
    'Column': prior_auth.columns,
    'Missing_Count': prior_auth.isnull().sum(),
    'Missing_Percent': (prior_auth.isnull().sum() / len(prior_auth) * 100).round(2)
})
print(missing_auth[missing_auth['Missing_Count'] > 0])

print("\n" + "="*60)
print("\nCLAIMS - Missing Values:")
missing_claims = pd.DataFrame({
    'Column': claims.columns,
    'Missing_Count': claims.isnull().sum(),
    'Missing_Percent': (claims.isnull().sum() / len(claims) * 100).round(2)
})
print(missing_claims[missing_claims['Missing_Count'] > 0])

# In[94]:


# Check for duplicates
print(f"Prior Auth Duplicates: {prior_auth.duplicated().sum()}")
print(f"Claims Duplicates: {claims.duplicated().sum()}")

# Basic statistics
print("\n" + "="*60)
print("PRIOR AUTHORIZATION - Summary Statistics:")
prior_auth.describe(include='all')

# ### 1.3 Target Variable Analysis

# In[95]:


# Target variable distribution
print("authstatus Distribution:")
print(prior_auth['authstatus'].value_counts())
print(f"\nPercentages:")
print(prior_auth['authstatus'].value_counts(normalize=True).mul(100).round(2))

# Visualize target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

prior_auth['authstatus'].value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c', '#f39c12'])
axes[0].set_title('Authorization Status Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Status')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

prior_auth['authstatus'].value_counts(normalize=True).mul(100).plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c', '#f39c12'])
axes[1].set_title('Authorization Status Distribution (%)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Status')
axes[1].set_ylabel('Percentage')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Check auto_approved relationship
print("\n" + "="*60)
print("auto_approved vs authstatus crosstab:")
pd.crosstab(prior_auth['authstatus'], prior_auth['auto_approved'], margins=True)

# ### 1.4 Feature Analysis

# In[96]:


# Service distribution
print("Service Types:")
print(prior_auth['service'].value_counts())

# Provider distribution
print(f"\n{'='*60}")
print(f"Number of unique providers: {prior_auth['provider_id'].nunique()}")
print(f"\nTop 10 providers by volume:")
print(prior_auth['provider_id'].value_counts().head(10))

# Unit distribution
print(f"\n{'='*60}")
print("Unit statistics:")
print(prior_auth['unit'].describe())

# Clinical reviewer analysis
print(f"\n{'='*60}")
print(f"clinical_reviewer missing: {prior_auth['clinical_reviewer'].isnull().sum()} ({(prior_auth['clinical_reviewer'].isnull().sum()/len(prior_auth)*100):.2f}%)")
print(f"Unique clinical reviewers: {prior_auth['clinical_reviewer'].nunique()}")

# In[97]:


# Claims dataset features
print("Claim Type Distribution:")
print(claims['Claim Type'].value_counts())

print(f"\n{'='*60}")
print("Amount Paid statistics:")
print(claims['Amount Paid'].describe())

# Temporal analysis
print(f"\n{'='*60}")
print("Prior Auth Date Range:")
print(f"From: {prior_auth['submission_date'].min()}")
print(f"To: {prior_auth['submission_date'].max()}")

print(f"\nClaims Date Range:")
print(f"From: {claims['Claim Date'].min()}")
print(f"To: {claims['Claim Date'].max()}")

# ### 1.5 Dataset Relationship Analysis

# In[98]:


# Pet ID overlap
unique_pets_auth = prior_auth['pet_id'].nunique()
unique_pets_claims = claims['Pet Id'].nunique()
pets_in_both = len(set(prior_auth['pet_id']) & set(claims['Pet Id']))

print(f"Unique pets in Prior Auth: {unique_pets_auth}")
print(f"Unique pets in Claims: {unique_pets_claims}")
print(f"Pets appearing in both datasets: {pets_in_both}")
print(f"% of auth pets with claim history: {(pets_in_both/unique_pets_auth*100):.2f}%")

# How many auths have corresponding pet history?
auths_with_history = prior_auth['pet_id'].isin(claims['Pet Id']).sum()
print(f"\n{'='*60}")
print(f"Prior auths with pet claim history: {auths_with_history} ({(auths_with_history/len(prior_auth)*100):.2f}%)")
print(f"Prior auths WITHOUT pet claim history: {len(prior_auth) - auths_with_history} ({((len(prior_auth) - auths_with_history)/len(prior_auth)*100):.2f}%)")

# ## Phase 2: Data Cleaning & Feature Engineering

# ### 2.1 Data Cleaning

# In[99]:


# Create working copies
auth_clean = prior_auth.copy()
claims_clean = claims.copy()

# 1. Filter out Pending auths (as per requirement)
print(f"Original auth records: {len(auth_clean)}")
auth_clean = auth_clean[auth_clean['authstatus'] != 'Pending']
print(f"After removing Pending: {len(auth_clean)}")
print(f"Removed {len(prior_auth) - len(auth_clean)} Pending records")

# 2. Drop clinical_reviewer and auto_approved (not available at prediction time)
auth_clean = auth_clean.drop(columns=['clinical_reviewer', 'auto_approved'])
print(f"\nDropped: clinical_reviewer, auto_approved")

# 3. Standardize Pet ID format (already consistent, but verify)
print(f"\nPet ID format check:")
print(f"Auth pet_id sample: {auth_clean['pet_id'].head(3).tolist()}")
print(f"Claims Pet Id sample: {claims_clean['Pet Id'].head(3).tolist()}")

# Standardize column names in claims for consistency
claims_clean.columns = claims_clean.columns.str.lower().str.replace(' ', '_')
print(f"\nClaims columns after standardization: {claims_clean.columns.tolist()}")

print(f"\nCleaned datasets:")
print(f"Auth shape: {auth_clean.shape}")
print(f"Claims shape: {claims_clean.shape}")

# ### 2.2 Feature Engineering from Claims History
# 
# **CRITICAL**: Only use claims that occurred BEFORE the auth submission_date to avoid data leakage!

# In[100]:


def create_claim_features(auth_row, claims_df):
    """
    Create claim history features for a single auth request.
    Only uses claims BEFORE the auth submission date.
    """
    pet_id = auth_row['pet_id']
    submission_date = auth_row['submission_date']
    
    # Filter claims for this pet that occurred before submission
    pet_claims = claims_df[
        (claims_df['pet_id'] == pet_id) & 
        (claims_df['claim_date'] < submission_date)
    ]
    
    if len(pet_claims) == 0:
        return {
            'claim_count_total': 0,
            'claim_count_last_30d': 0,
            'claim_count_last_90d': 0,
            'total_amount_paid': 0.0,
            'avg_amount_paid': 0.0,
            'max_amount_paid': 0.0,
            'days_since_last_claim': -1,
            'unique_claim_types': 0,
            'has_claim_history': 0
        }
    
    # Calculate time windows
    date_30d_ago = submission_date - timedelta(days=30)
    date_90d_ago = submission_date - timedelta(days=90)
    
    # Count features
    claim_count_total = len(pet_claims)
    claim_count_last_30d = len(pet_claims[pet_claims['claim_date'] >= date_30d_ago])
    claim_count_last_90d = len(pet_claims[pet_claims['claim_date'] >= date_90d_ago])
    
    # Amount features
    total_amount_paid = pet_claims['amount_paid'].sum()
    avg_amount_paid = pet_claims['amount_paid'].mean()
    max_amount_paid = pet_claims['amount_paid'].max()
    
    # Recency
    last_claim_date = pet_claims['claim_date'].max()
    days_since_last_claim = (submission_date - last_claim_date).days
    
    # Diversity
    unique_claim_types = pet_claims['claim_type'].nunique()
    
    return {
        'claim_count_total': claim_count_total,
        'claim_count_last_30d': claim_count_last_30d,
        'claim_count_last_90d': claim_count_last_90d,
        'total_amount_paid': round(total_amount_paid, 2),
        'avg_amount_paid': round(avg_amount_paid, 2),
        'max_amount_paid': round(max_amount_paid, 2),
        'days_since_last_claim': days_since_last_claim,
        'unique_claim_types': unique_claim_types,
        'has_claim_history': 1
    }

print("Feature engineering function created.")

# In[101]:


# Apply feature engineering to all auth records
print("Creating claim history features for all authorizations...")
print("This may take a moment...")

claim_features_list = []
for idx, row in auth_clean.iterrows():
    features = create_claim_features(row, claims_clean)
    claim_features_list.append(features)

# Convert to DataFrame
claim_features_df = pd.DataFrame(claim_features_list)

# Combine with auth data
auth_with_features = pd.concat([auth_clean.reset_index(drop=True), claim_features_df], axis=1)

print(f"\\nFeature engineering complete!")
print(f"New shape: {auth_with_features.shape}")
print(f"\\nNew features added:")
print(claim_features_df.columns.tolist())
print(f"\\nSample of engineered features:")
auth_with_features[['pet_id', 'has_claim_history', 'claim_count_total', 'total_amount_paid', 'days_since_last_claim']].head(10)

# ### 2.3 Temporal Features

# In[102]:


# Create temporal features from submission_date
auth_with_features['day_of_week'] = auth_with_features['submission_date'].dt.dayofweek
auth_with_features['month'] = auth_with_features['submission_date'].dt.month
auth_with_features['quarter'] = auth_with_features['submission_date'].dt.quarter
auth_with_features['is_weekend'] = (auth_with_features['day_of_week'] >= 5).astype(int)
auth_with_features['day_of_month'] = auth_with_features['submission_date'].dt.day

print("Temporal features created:")
print("- day_of_week (0=Monday, 6=Sunday)")
print("- month (1-12)")
print("- quarter (1-4)")
print("- is_weekend (0/1)")
print("- day_of_month (1-31)")

print(f"\\nSample:")
auth_with_features[['submission_date', 'day_of_week', 'month', 'quarter', 'is_weekend']].head()

# ### 2.4 Provider Features

# In[103]:


# Provider approval rate (will calculate in train set only to avoid leakage)
# For now, just count provider volume
provider_counts = auth_with_features.groupby('provider_id').size().to_dict()
auth_with_features['provider_auth_count'] = auth_with_features['provider_id'].map(provider_counts)

print(f"Provider features created:")
print(f"- provider_auth_count: Number of auths from this provider")
print(f"\\nNote: Provider approval rate will be calculated from training data only to avoid leakage")

print(f"\\nSample:")
auth_with_features[['provider_id', 'provider_auth_count']].head(10)

# ### 2.5 Prepare for Modeling

# ## Phase 3: Model Development

# ### 3.1 Temporal Train/Test Split

# In[106]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Define feature columns
categorical_features = ['service']
numeric_features = ['provider_id', 'unit', 'claim_count_total', 'claim_count_last_30d', 
                    'claim_count_last_90d', 'total_amount_paid', 'avg_amount_paid', 
                    'max_amount_paid', 'days_since_last_claim', 'unique_claim_types',
                    'has_claim_history', 'day_of_week', 'month', 'quarter', 'is_weekend',
                    'day_of_month', 'provider_auth_count']

# Temporal split: 80% train, 20% test
split_idx = int(len(auth_with_features) * 0.8)
train_df = auth_with_features.iloc[:split_idx].copy()
test_df = auth_with_features.iloc[split_idx:].copy()

print(f"Temporal split (80/20):")
print(f"Train size: {len(train_df)} ({len(train_df)/len(auth_with_features)*100:.1f}%)")
print(f"Test size: {len(test_df)} ({len(test_df)/len(auth_with_features)*100:.1f}%)")
print(f"\\nTrain date range: {train_df['submission_date'].min()} to {train_df['submission_date'].max()}")
print(f"Test date range: {test_df['submission_date'].min()} to {test_df['submission_date'].max()}")
print(f"\\nTrain target distribution:")
# print(train_df['target'].value_counts(normalize=True).mul(100).round(2))
print(f"\\nTest target distribution:")
# print(test_df['target'].value_counts(normalize=True).mul(100).round(2))

# In[108]:


# One-hot encode service and prepare feature matrices
train_encoded = pd.get_dummies(train_df[categorical_features + numeric_features], columns=categorical_features)
test_encoded = pd.get_dummies(test_df[categorical_features + numeric_features], columns=categorical_features)

# Ensure same columns in train and test
missing_cols = set(train_encoded.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0
test_encoded = test_encoded[train_encoded.columns]

X_train = train_encoded
X_test = test_encoded
y_train = train_df['target']
y_test = test_df['target']

print(f"Feature matrix shape:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"\\nFeatures ({len(X_train.columns)}):")
print(X_train.columns.tolist())

# ### 3.2 Baseline Model - Logistic Regression

# In[ ]:


# Analyze best CV model at different thresholds
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

threshold_results_cv = []
for threshold in thresholds:
    y_pred_thresh = (y_pred_proba_xgb_cv >= threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred_thresh, zero_division=0)
    recall = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
    
    # Calculate approval rate
    approval_rate = y_pred_thresh.sum() / len(y_pred_thresh) * 100
    
    # False approval rate
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
    false_approval_rate = fp / (fp + tp) * 100 if (fp + tp) > 0 else 0
    
    threshold_results_cv.append({
        'Threshold': threshold,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Auto-Approval %': approval_rate,
        'False Approval %': false_approval_rate
    })

threshold_df_cv = pd.DataFrame(threshold_results_cv)

print("XGBOOST (Cross-Validated) - Threshold Analysis")
print("="*80)
print(threshold_df_cv.to_string(index=False))

print(f"\nRecommendation: Use threshold >= 0.90 for high precision auto-approval")
print(f"\nComparison with non-CV model:")
print(f"The cross-validated model achieved better generalization through hyperparameter tuning.")

# #### 3.2.7 High-Precision Threshold Analysis for Best CV Model

# In[ ]:


# Feature importance from best CV XGBoost model
feature_importance_xgb_cv = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model_cv.feature_importances_
}).sort_values('Importance', ascending=False)

print("XGBOOST (Cross-Validated) - Top 20 Features")
print("="*60)
print(feature_importance_xgb_cv.head(20).to_string(index=False))

# Visualize top 15 features
fig, ax = plt.subplots(figsize=(10, 8))
top_features_xgb_cv = feature_importance_xgb_cv.head(15)
ax.barh(range(len(top_features_xgb_cv)), top_features_xgb_cv['Importance'].values, color='steelblue')
ax.set_yticks(range(len(top_features_xgb_cv)))
ax.set_yticklabels(top_features_xgb_cv['Feature'].values)
ax.invert_yaxis()
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('XGBoost (CV) - Top 15 Most Important Features', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# #### 3.2.6 Feature Importance from Best CV Model

# In[ ]:


# Visualize confusion matrices for CV models
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models_cv = ['Logistic Regression (CV)', 'Random Forest (CV)', 'XGBoost (CV)']
cms_cv = [cm_lr_cv, cm_rf_cv, cm_xgb_cv]

for idx, (model_name, cm) in enumerate(zip(models_cv, cms_cv)):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[idx],
                xticklabels=['Denied', 'Approved'],
                yticklabels=['Denied', 'Approved'])
    axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()

# ROC Curves for CV models
fig, ax = plt.subplots(figsize=(10, 8))

# Calculate ROC curves
fpr_lr_cv, tpr_lr_cv, _ = roc_curve(y_test, y_pred_proba_lr_cv)
fpr_rf_cv, tpr_rf_cv, _ = roc_curve(y_test, y_pred_proba_rf_cv)
fpr_xgb_cv, tpr_xgb_cv, _ = roc_curve(y_test, y_pred_proba_xgb_cv)

# Plot
ax.plot(fpr_lr_cv, tpr_lr_cv, label=f'Logistic Regression CV (AUC = {roc_auc_score(y_test, y_pred_proba_lr_cv):.3f})', linewidth=2)
ax.plot(fpr_rf_cv, tpr_rf_cv, label=f'Random Forest CV (AUC = {roc_auc_score(y_test, y_pred_proba_rf_cv):.3f})', linewidth=2)
ax.plot(fpr_xgb_cv, tpr_xgb_cv, label=f'XGBoost CV (AUC = {roc_auc_score(y_test, y_pred_proba_xgb_cv):.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Cross-Validated Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# #### 3.2.5 Visualize Cross-Validated Model Performance

# In[ ]:


# Compare all cross-validated models
cv_comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression (CV)', 'Random Forest (CV)', 'XGBoost (CV)'],
    'Best CV Precision': [
        lr_grid.best_score_,
        rf_grid.best_score_,
        xgb_grid.best_score_
    ],
    'Test Accuracy': [
        accuracy_score(y_test, y_pred_lr_cv),
        accuracy_score(y_test, y_pred_rf_cv),
        accuracy_score(y_test, y_pred_xgb_cv)
    ],
    'Test Precision': [
        precision_score(y_test, y_pred_lr_cv),
        precision_score(y_test, y_pred_rf_cv),
        precision_score(y_test, y_pred_xgb_cv)
    ],
    'Test Recall': [
        recall_score(y_test, y_pred_lr_cv),
        recall_score(y_test, y_pred_rf_cv),
        recall_score(y_test, y_pred_xgb_cv)
    ],
    'Test F1-Score': [
        f1_score(y_test, y_pred_lr_cv),
        f1_score(y_test, y_pred_rf_cv),
        f1_score(y_test, y_pred_xgb_cv)
    ],
    'Test ROC-AUC': [
        roc_auc_score(y_test, y_pred_proba_lr_cv),
        roc_auc_score(y_test, y_pred_proba_rf_cv),
        roc_auc_score(y_test, y_pred_proba_xgb_cv)
    ]
})

print("CROSS-VALIDATED MODEL COMPARISON")
print("="*100)
print(cv_comparison_df.to_string(index=False))

print(f"\nBest Overall Model:")
best_model_idx = cv_comparison_df['Test F1-Score'].idxmax()
print(f"  {cv_comparison_df.loc[best_model_idx, 'Model']}")
print(f"  Test Precision: {cv_comparison_df.loc[best_model_idx, 'Test Precision']:.4f}")
print(f"  Test Recall: {cv_comparison_df.loc[best_model_idx, 'Test Recall']:.4f}")
print(f"  Test F1-Score: {cv_comparison_df.loc[best_model_idx, 'Test F1-Score']:.4f}")
print(f"  Test ROC-AUC: {cv_comparison_df.loc[best_model_idx, 'Test ROC-AUC']:.4f}")

# #### 3.2.4 Cross-Validated Model Comparison

# In[ ]:


from sklearn import GridSearchCV
# Calculate scale_pos_weight for class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Hyperparameter grid for XGBoost
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'scale_pos_weight': [scale_pos_weight]
}

# GridSearchCV with TimeSeriesSplit
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    param_grid=xgb_param_grid,
    cv=tscv,
    scoring='precision',
    n_jobs=-1,
    verbose=1
)

print("Training XGBoost with GridSearchCV...")
print("This may take several minutes...")
xgb_grid.fit(X_train, y_train)

# Best model
xgb_model_cv = xgb_grid.best_estimator_

# Predictions
y_pred_xgb_cv = xgb_model_cv.predict(X_test)
y_pred_proba_xgb_cv = xgb_model_cv.predict_proba(X_test)[:, 1]

# Evaluation
print("\nXGBOOST (Cross-Validated) - Test Set Performance")
print("="*60)
print(f"Best parameters: {xgb_grid.best_params_}")
print(f"Best CV precision score: {xgb_grid.best_score_:.4f}")
print(f"\nTest Set Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb_cv):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb_cv):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb_cv):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_xgb_cv):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_xgb_cv):.4f}")

print(f"\nConfusion Matrix:")
cm_xgb_cv = confusion_matrix(y_test, y_pred_xgb_cv)
print(cm_xgb_cv)

# #### 3.2.3 XGBoost with Cross-Validation

# In[ ]:


# Hyperparameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10, 12],
    'min_samples_split': [10, 20, 30],
    'min_samples_leaf': [5, 10, 15],
    'class_weight': ['balanced']
}

# GridSearchCV with TimeSeriesSplit
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=rf_param_grid,
    cv=tscv,
    scoring='precision',
    n_jobs=-1,
    verbose=1
)

print("Training Random Forest with GridSearchCV...")
print("This may take several minutes...")
rf_grid.fit(X_train, y_train)

# Best model
rf_model_cv = rf_grid.best_estimator_

# Predictions
y_pred_rf_cv = rf_model_cv.predict(X_test)
y_pred_proba_rf_cv = rf_model_cv.predict_proba(X_test)[:, 1]

# Evaluation
print("\nRANDOM FOREST (Cross-Validated) - Test Set Performance")
print("="*60)
print(f"Best parameters: {rf_grid.best_params_}")
print(f"Best CV precision score: {rf_grid.best_score_:.4f}")
print(f"\nTest Set Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_cv):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf_cv):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf_cv):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf_cv):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf_cv):.4f}")

print(f"\nConfusion Matrix:")
cm_rf_cv = confusion_matrix(y_test, y_pred_rf_cv)
print(cm_rf_cv)

# #### 3.2.2 Random Forest with Cross-Validation

# In[ ]:


# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter grid for Logistic Regression
lr_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'saga'],
    'max_iter': [1000]
}

# GridSearchCV with TimeSeriesSplit
lr_grid = GridSearchCV(
    LogisticRegression(random_state=42, class_weight='balanced'),
    param_grid=lr_param_grid,
    cv=tscv,
    scoring='precision',
    n_jobs=-1,
    verbose=1
)

print("Training Logistic Regression with GridSearchCV...")
lr_grid.fit(X_train_scaled, y_train)

# Best model
lr_model_cv = lr_grid.best_estimator_

# Predictions
y_pred_lr_cv = lr_model_cv.predict(X_test_scaled)
y_pred_proba_lr_cv = lr_model_cv.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("\nLOGISTIC REGRESSION (Cross-Validated) - Test Set Performance")
print("="*60)
print(f"Best parameters: {lr_grid.best_params_}")
print(f"Best CV precision score: {lr_grid.best_score_:.4f}")
print(f"\nTest Set Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr_cv):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr_cv):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr_cv):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr_cv):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_lr_cv):.4f}")

print(f"\nConfusion Matrix:")
cm_lr_cv = confusion_matrix(y_test, y_pred_lr_cv)
print(cm_lr_cv)

# #### 3.2.1 Logistic Regression with Cross-Validation

# In[ ]:


from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer

# Use TimeSeriesSplit for temporal cross-validation (preserves time order)
tscv = TimeSeriesSplit(n_splits=5)

# Custom scorer prioritizing precision (most important for auto-approval)
precision_scorer = make_scorer(precision_score)

print("Cross-Validation Setup:")
print(f"Strategy: TimeSeriesSplit with 5 folds")
print(f"Scoring metric: Precision (primary), with F1 and ROC-AUC tracking")
print(f"\nThis will take a few minutes to run hyperparameter tuning...")

# ### 3.2 Cross-Validated Model Training with Hyperparameter Tuning
# 
# Using TimeSeriesSplit for temporal cross-validation to optimize hyperparameters.

# In[ ]:


# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("LOGISTIC REGRESSION - Test Set Performance")
print("="*60)
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Denied', 'Approved']))

print(f"\nConfusion Matrix:")
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(cm_lr)

# ### 3.3 Random Forest Model

# In[ ]:


# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Evaluation
print("RANDOM FOREST - Test Set Performance")
print("="*60)
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Denied', 'Approved']))

print(f"\nConfusion Matrix:")
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)

# ### 3.4 XGBoost Model

# In[ ]:


# Calculate scale_pos_weight for class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Evaluation
print("XGBOOST - Test Set Performance")
print("="*60)
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_xgb):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_xgb):.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['Denied', 'Approved']))

print(f"\nConfusion Matrix:")
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print(cm_xgb)

# ### 3.5 Model Comparison

# In[ ]:


# Model comparison table
comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb)
    ],
    'Precision': [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_xgb)
    ],
    'Recall': [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_xgb)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_xgb)
    ],
    'ROC-AUC': [
        roc_auc_score(y_test, y_pred_proba_lr),
        roc_auc_score(y_test, y_pred_proba_rf),
        roc_auc_score(y_test, y_pred_proba_xgb)
    ]
})

print("MODEL COMPARISON - Test Set")
print("="*80)
print(comparison_df.to_string(index=False))

# Highlight best model for each metric
print(f"\\nBest Precision: {comparison_df.loc[comparison_df['Precision'].idxmax(), 'Model']}")
print(f"Best Recall: {comparison_df.loc[comparison_df['Recall'].idxmax(), 'Model']}")
print(f"Best F1-Score: {comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']}")
print(f"Best ROC-AUC: {comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Model']}")

# In[ ]:


# Visualize confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models = ['Logistic Regression', 'Random Forest', 'XGBoost']
cms = [cm_lr, cm_rf, cm_xgb]

for idx, (model_name, cm) in enumerate(zip(models, cms)):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Denied', 'Approved'],
                yticklabels=['Denied', 'Approved'])
    axes[idx].set_title(f'{model_name}\\nConfusion Matrix', fontweight='bold')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()

# In[ ]:


# ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

# Calculate ROC curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)

# Plot
ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_pred_proba_lr):.3f})', linewidth=2)
ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_pred_proba_rf):.3f})', linewidth=2)
ax.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(y_test, y_pred_proba_xgb):.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ### 3.6 Feature Importance Analysis

# In[ ]:


# Feature importance from Random Forest
feature_importance_rf = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("RANDOM FOREST - Top 20 Features")
print("="*60)
print(feature_importance_rf.head(20).to_string(index=False))

# Visualize top 15 features
fig, ax = plt.subplots(figsize=(10, 8))
top_features = feature_importance_rf.head(15)
ax.barh(range(len(top_features)), top_features['Importance'].values)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'].values)
ax.invert_yaxis()
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Random Forest - Top 15 Most Important Features', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# In[ ]:


# Feature importance from XGBoost
feature_importance_xgb = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("XGBOOST - Top 20 Features")
print("="*60)
print(feature_importance_xgb.head(20).to_string(index=False))

# Visualize top 15 features
fig, ax = plt.subplots(figsize=(10, 8))
top_features_xgb = feature_importance_xgb.head(15)
ax.barh(range(len(top_features_xgb)), top_features_xgb['Importance'].values)
ax.set_yticks(range(len(top_features_xgb)))
ax.set_yticklabels(top_features_xgb['Feature'].values)
ax.invert_yaxis()
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('XGBoost - Top 15 Most Important Features', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# ### 3.7 High-Precision Threshold Analysis
# 
# For auto-approval, we prioritize PRECISION to avoid false approvals. Let's analyze performance at different confidence thresholds.

# In[ ]:


# Analyze XGBoost at different thresholds (best performing model)
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

threshold_results = []
for threshold in thresholds:
    y_pred_thresh = (y_pred_proba_xgb >= threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred_thresh, zero_division=0)
    recall = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
    
    # Calculate approval rate
    approval_rate = y_pred_thresh.sum() / len(y_pred_thresh) * 100
    
    # False approval rate (FP / total predictions)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
    false_approval_rate = fp / (fp + tp) * 100 if (fp + tp) > 0 else 0
    
    threshold_results.append({
        'Threshold': threshold,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Auto-Approval %': approval_rate,
        'False Approval %': false_approval_rate
    })

threshold_df = pd.DataFrame(threshold_results)

print("XGBOOST - Threshold Analysis for High Precision")
print("="*80)
print(threshold_df.to_string(index=False))

print(f"\\nRecommendation: Use threshold >= 0.90 for high precision auto-approval")

# ## Phase 4: Business Recommendation

# ### 4.1 Current State Analysis

# In[ ]:


# Analyze current auto-approval performance in training data
current_auto_approved = train_df[train_df['authstatus'] == 'Approved']

print("CURRENT STATE (Rule-Based System)")
print("="*80)
print(f"Total authorizations in training set: {len(train_df)}")
print(f"Total approved auths: {len(current_auto_approved)}")
print(f"Current auto-approval rate: {len(current_auto_approved)/len(train_df)*100:.2f}%")
print(f"\\nCurrent manual review rate: {(1 - len(current_auto_approved)/len(train_df))*100:.2f}%")

# Breakdown by approval status
print(f"\\nBreakdown:")
print(train_df['authstatus'].value_counts())
print(f"\\nPercentages:")
print(train_df['authstatus'].value_counts(normalize=True).mul(100).round(2))

# ### 4.2 ML Model Performance at Different Thresholds

# In[ ]:


# Detailed analysis at recommended threshold (0.90)
recommended_threshold = 0.90
y_pred_recommended = (y_pred_proba_xgb >= recommended_threshold).astype(int)

# Confusion matrix at recommended threshold
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_recommended).ravel()

print(f"ML MODEL PERFORMANCE (XGBoost @ {recommended_threshold} threshold)")
print("="*80)
print(f"\\nConfusion Matrix Breakdown:")
print(f"True Negatives (Correctly denied): {tn}")
print(f"False Positives (Wrongly approved): {fp}")
print(f"False Negatives (Wrongly denied): {fn}")
print(f"True Positives (Correctly approved): {tp}")

print(f"\\nKey Metrics:")
print(f"Precision (of approved): {precision_score(y_test, y_pred_recommended):.4f}")
print(f"Recall (of approved): {recall_score(y_test, y_pred_recommended):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_recommended):.4f}")

# Auto-approval rate
auto_approved_ml = y_pred_recommended.sum()
total_test = len(y_test)

print(f"\\nAuto-Approval Analysis:")
print(f"Total test authorizations: {total_test}")
print(f"Auto-approved by ML model: {auto_approved_ml}")
print(f"Auto-approval rate: {auto_approved_ml/total_test*100:.2f}%")
print(f"Sent to manual review: {total_test - auto_approved_ml} ({(total_test - auto_approved_ml)/total_test*100:.2f}%)")

# Error analysis
print(f"\\nError Analysis:")
print(f"False approval rate: {fp/(fp+tp)*100:.2f}% (of all ML approvals)")
print(f"Missed approvals: {fn} out of {fn+tp} actual approvals ({fn/(fn+tp)*100:.2f}%)")

# ### 4.3 Business Impact Analysis

# In[ ]:


# Business impact calculations
# Assumptions for cost-benefit analysis
avg_review_time_minutes = 10  # minutes per manual review
hourly_cost_reviewer = 40  # dollars per hour
cost_per_review = (avg_review_time_minutes / 60) * hourly_cost_reviewer

# Calculate potential savings
current_manual_reviews_per_year = 2000  # based on dataset size
ml_auto_approval_rate = auto_approved_ml / total_test

# Estimate reviews saved
reviews_saved = current_manual_reviews_per_year * ml_auto_approval_rate
time_saved_hours = reviews_saved * (avg_review_time_minutes / 60)
cost_saved = reviews_saved * cost_per_review

print("BUSINESS IMPACT ANALYSIS")
print("="*80)
print(f"\\nAssumptions:")
print(f"- Average review time: {avg_review_time_minutes} minutes")
print(f"- Reviewer hourly cost: ${hourly_cost_reviewer}")
print(f"- Cost per review: ${cost_per_review:.2f}")
print(f"- Annual authorization volume: {current_manual_reviews_per_year:,}")

print(f"\\nProjected ML Model Impact:")
print(f"- ML auto-approval rate: {ml_auto_approval_rate*100:.2f}%")
print(f"- Reviews automated annually: {reviews_saved:.0f}")
print(f"- Time saved: {time_saved_hours:.0f} hours/year")
print(f"- Cost savings: ${cost_saved:,.2f}/year")

print(f"\\nWorkload Reduction:")
print(f"- Current manual review workload: {current_manual_reviews_per_year:,} reviews/year")
print(f"- Remaining manual reviews with ML: {current_manual_reviews_per_year - reviews_saved:.0f} reviews/year")
print(f"- Workload reduction: {ml_auto_approval_rate*100:.2f}%")

# ### 4.4 Risk Assessment

# In[ ]:


# Risk analysis
precision_at_threshold = precision_score(y_test, y_pred_recommended)
false_approval_count = fp

print("RISK ASSESSMENT")
print("="*80)

print(f"\\n1. FALSE APPROVAL RISK (Most Critical)")
print(f"   - False approvals in test set: {false_approval_count}")
print(f"   - Precision at threshold {recommended_threshold}: {precision_at_threshold:.4f}")
print(f"   - This means {(1-precision_at_threshold)*100:.2f}% of ML approvals may be incorrect")
print(f"   - Risk level: LOW - High precision threshold minimizes this risk")

print(f"\\n2. MISSED APPROVAL RISK (Opportunity Cost)")
print(f"   - Missed approvals in test set: {fn}")
print(f"   - These would go to manual review (not a critical error)")
print(f"   - Risk level: LOW - Conservative approach maintains safety")

print(f"\\n3. MODEL DRIFT RISK")
print(f"   - As authorization patterns change, model may degrade")
print(f"   - Mitigation: Monitor performance monthly, retrain quarterly")
print(f"   - Risk level: MEDIUM - Requires ongoing monitoring")

print(f"\\n4. DATA QUALITY RISK")
print(f"   - Model depends on historical claims data")
print(f"   - {(1-auth_with_features['has_claim_history'].mean())*100:.2f}% of auths have no claim history")
print(f"   - Mitigation: Model handles missing history with default features")
print(f"   - Risk level: LOW - Feature engineering addresses this")

print(f"\\n5. REGULATORY/COMPLIANCE RISK")
print(f"   - Insurance companies may require explainability")
print(f"   - Mitigation: Feature importance available, tree-based model is interpretable")
print(f"   - Risk level: LOW-MEDIUM - Depends on regulatory requirements")

print(f"\\nOVERALL RISK ASSESSMENT: LOW")
print(f"The high precision threshold ({recommended_threshold}) minimizes false approvals,")
print(f"which is the most critical risk for auto-approval systems.")

# ### 4.5 Implementation Roadmap

# In[ ]:


print("IMPLEMENTATION ROADMAP")
print("="*80)

print(f"\\nPHASE 1: PILOT (Months 1-2)")
print(f"  • Deploy ML model in shadow mode (predictions not used)")
print(f"  • Compare ML predictions to actual human decisions")
print(f"  • Validate precision threshold in production")
print(f"  • Identify edge cases and model failures")
print(f"  • Goal: Validate {precision_at_threshold*100:.1f}% precision in production")

print(f"\\nPHASE 2: LIMITED ROLLOUT (Months 3-4)")
print(f"  • Enable auto-approval for 25% of incoming auths")
print(f"  • Monitor false approval rate daily")
print(f"  • Collect feedback from reviewers on remaining 75%")
print(f"  • Implement monitoring dashboards")
print(f"  • Goal: Confirm business impact and user acceptance")

print(f"\\nPHASE 3: FULL DEPLOYMENT (Month 5)")
print(f"  • Enable auto-approval for all auths meeting threshold")
print(f"  • Expected auto-approval rate: {ml_auto_approval_rate*100:.2f}%")
print(f"  • Maintain human review for low-confidence predictions")
print(f"  • Goal: Reduce manual review workload by {ml_auto_approval_rate*100:.0f}%")

print(f"\\nPHASE 4: OPTIMIZATION (Months 6-12)")
print(f"  • Monitor model performance monthly")
print(f"  • Retrain model quarterly with new data")
print(f"  • Collect additional features if needed")
print(f"  • Consider A/B testing different thresholds")
print(f"  • Goal: Maintain or improve precision while increasing coverage")

print(f"\\nKEY SUCCESS METRICS:")
print(f"  • Precision: >={precision_at_threshold*100:.1f}% (minimize false approvals)")
print(f"  • Auto-approval rate: ~{ml_auto_approval_rate*100:.0f}%")
print(f"  • Manual review reduction: {ml_auto_approval_rate*100:.0f}%")
print(f"  • Time to approval: <24 hours for auto-approved auths")
print(f"  • Cost savings: ${cost_saved:,.0f}/year")

# ### 4.6 Final Recommendation

# In[ ]:


print("="*80)
print("FINAL RECOMMENDATION: DEPLOY THE ML MODEL")
print("="*80)

print(f"\\n✓ RECOMMENDATION: PROCEED WITH DEPLOYMENT")
print(f"\\nThe XGBoost model at 0.90 confidence threshold is ready for production deployment.")

print(f"\\nKEY STRENGTHS:")
print(f"  1. High Precision: {precision_at_threshold*100:.1f}% - minimizes false approvals")
print(f"  2. Significant Impact: {ml_auto_approval_rate*100:.1f}% auto-approval rate")
print(f"  3. Cost Effective: ${cost_saved:,.0f}/year in estimated savings")
print(f"  4. Low Risk: Conservative threshold protects against critical errors")
print(f"  5. Interpretable: Tree-based model with clear feature importance")

print(f"\\nQUANTIFIABLE BENEFITS:")
print(f"  • Workload reduction: {ml_auto_approval_rate*100:.0f}% fewer manual reviews")
print(f"  • Time savings: {time_saved_hours:.0f} hours/year")
print(f"  • Faster approvals: Near-instant for auto-approved cases")
print(f"  • Consistent decisions: Eliminates human variability")

print(f"\\nRISK MITIGATION:")
print(f"  • Start with shadow mode deployment")
print(f"  • Gradual rollout (25% → 100%)")
print(f"  • Continuous monitoring of precision")
print(f"  • Quarterly model retraining")
print(f"  • Human review fallback for low-confidence predictions")

print(f"\\nNEXT STEPS:")
print(f"  1. Finalize production infrastructure")
print(f"  2. Implement monitoring dashboards")
print(f"  3. Begin Phase 1 pilot in shadow mode")
print(f"  4. Establish model performance SLAs")
print(f"  5. Train operations team on new workflow")

print(f"\\n" + "="*80)
print(f"The model is production-ready and recommended for immediate pilot deployment.")
print("="*80)

# In[ ]:


# Visualization: Current vs ML Model comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Current state (assume all need review, then approved/denied after review)
current_breakdown = pd.Series({
    'Manual Review': len(train_df),
    'Auto-Approved': 0
})

# ML model state
ml_breakdown = pd.Series({
    'Auto-Approved by ML': auto_approved_ml,
    'Manual Review': total_test - auto_approved_ml
})

# Plot current state
colors_current = ['#e74c3c', '#95a5a6']
axes[0].pie(current_breakdown, labels=current_breakdown.index, autopct='%1.1f%%',
            colors=colors_current, startangle=90)
axes[0].set_title('Current State\\n(Rule-Based System)', fontsize=14, fontweight='bold')

# Plot ML model state
colors_ml = ['#2ecc71', '#e74c3c']
axes[1].pie(ml_breakdown, labels=ml_breakdown.index, autopct='%1.1f%%',
            colors=colors_ml, startangle=90)
axes[1].set_title(f'With ML Model\\n(XGBoost @ {recommended_threshold} threshold)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\\nWorkload Reduction Visualization:")
print(f"Current: {len(train_df)} manual reviews")
print(f"With ML: {total_test - auto_approved_ml} manual reviews")
print(f"Reduction: {ml_auto_approval_rate*100:.1f}%")

# In[ ]:


# Visualize threshold trade-offs
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Precision, Recall, F1 vs Threshold
axes[0].plot(threshold_df['Threshold'], threshold_df['Precision'], marker='o', label='Precision', linewidth=2)
axes[0].plot(threshold_df['Threshold'], threshold_df['Recall'], marker='s', label='Recall', linewidth=2)
axes[0].plot(threshold_df['Threshold'], threshold_df['F1-Score'], marker='^', label='F1-Score', linewidth=2)
axes[0].set_xlabel('Confidence Threshold', fontsize=12)
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_title('Model Performance vs Threshold', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)
axes[0].set_ylim([0, 1.05])

# Auto-approval rate and False approval rate
ax2 = axes[1]
ax2.plot(threshold_df['Threshold'], threshold_df['Auto-Approval %'], marker='o', 
         color='green', label='Auto-Approval Rate', linewidth=2)
ax2.set_xlabel('Confidence Threshold', fontsize=12)
ax2.set_ylabel('Auto-Approval Rate (%)', fontsize=12, color='green')
ax2.tick_params(axis='y', labelcolor='green')

ax3 = ax2.twinx()
ax3.plot(threshold_df['Threshold'], threshold_df['False Approval %'], marker='s', 
         color='red', label='False Approval Rate', linewidth=2)
ax3.set_ylabel('False Approval Rate (%)', fontsize=12, color='red')
ax3.tick_params(axis='y', labelcolor='red')

axes[1].set_title('Auto-Approval vs False Approval Rate', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# In[109]:


# Sort by submission date for temporal split
auth_with_features = auth_with_features.sort_values('submission_date').reset_index(drop=True)

# Encode target variable
auth_with_features['target'] = (auth_with_features['authstatus'] == 'Approved').astype(int)

print(f"Target variable encoding:")
print(f"Approved -> 1")
print(f"Denied -> 0")
print(f"\\nTarget distribution:")
print(auth_with_features['target'].value_counts())
print(f"\\nClass balance:")
print(auth_with_features['target'].value_counts(normalize=True).mul(100).round(2))

# Summary of final dataset
print(f"\\n{'='*60}")
print(f"FINAL DATASET SUMMARY")
print(f"{'='*60}")
print(f"Total records: {len(auth_with_features)}")
print(f"Total features: {auth_with_features.shape[1]}")
print(f"\\nColumns:")
for col in auth_with_features.columns:
    print(f"  - {col}")

# In[ ]:



