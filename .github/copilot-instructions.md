# Copilot Instructions for Cohere Pets Prior Authorization ML Project

## Repository Overview

This repository contains a machine learning project for automating prior authorization decisions for veterinary care at Cohere Pets. The project uses historical prior authorization data and claims data to build predictive models that can auto-approve authorizations at submission time, reducing manual review workload and improving customer experience.

## Project Structure

- **`cohere_pets_analysis.ipynb`**: Main Jupyter notebook containing exploratory data analysis, feature engineering, model development, and evaluation for the prior authorization ML project
- **`llm_evaluation_framework.ipynb`**: Jupyter notebook with evaluation framework design for LLM-based clinical notes summarization chatbot
- **`Emma-_Data_Set.xlsx`**: Dataset containing prior authorization and claims data
- **`requirements.txt`**: Python dependencies for the project
- **`PROJECT_PLAN.md`**: Comprehensive project plan with phased approach, objectives, and deliverables
- **`FIXES_SUMMARY.md`**: Documentation of fixes and improvements made to the project

## Technology Stack

- **Python 3.x**: Primary programming language
- **Jupyter Notebooks**: Interactive development and analysis environment
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning models and evaluation
- **XGBoost**: Gradient boosting models for classification
- **imbalanced-learn**: Techniques for handling class imbalance (SMOTE)
- **matplotlib/seaborn**: Data visualization

## Working with Jupyter Notebooks

When modifying or creating Jupyter notebooks in this repository:

1. **Code Quality**: Ensure code in notebook cells is clean, well-commented, and follows Python best practices
2. **Cell Organization**: Keep cells focused on specific tasks; split long cells into logical chunks
3. **Markdown Documentation**: Use markdown cells liberally to explain analysis steps, findings, and decisions
4. **Output Management**: Clear outputs before committing if they contain sensitive data or are very large
5. **Reproducibility**: Set random seeds for reproducible results (e.g., `random_state=42`)
6. **Error Handling**: Include appropriate error handling and data validation in analysis code

## Data Science Best Practices

### Feature Engineering
- Only use information available at prediction time (avoid data leakage)
- Use temporal awareness when creating features from historical data
- Document feature creation logic with clear comments
- Consider features from both prior authorization and claims datasets

### Model Development
- Address class imbalance using SMOTE, class weights, or other appropriate techniques
- Use temporal train/test splits rather than random splits to prevent data leakage
- Optimize for business-relevant metrics (precision, recall, F1, F2-score) not just accuracy
- Implement threshold optimization based on business cost considerations
- Document model assumptions and limitations

### Evaluation
- Report comprehensive metrics: precision, recall, F1, F2, ROC-AUC, PR-AUC
- Analyze confusion matrices with focus on false positive and false negative rates
- Provide business impact analysis (workload reduction, cost-benefit)
- Consider multi-threshold strategies for different confidence levels

## Dependencies and Environment

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
- pandas==2.3.3
- numpy==2.3.5
- scikit-learn==1.5.2
- xgboost==3.1.2
- imbalanced-learn==0.12.4
- jupyter==1.1.1

## Code Style and Quality

- Follow PEP 8 style guidelines for Python code
- Use descriptive variable names (e.g., `approval_rate`, not `ar`)
- Add comments for complex logic or business-specific calculations
- Keep functions focused and single-purpose
- Use type hints where appropriate for function signatures

## Domain-Specific Guidelines

### Prior Authorization Context
- **authstatus**: Target variable with values (Approved/Denied/Pending)
- **Temporal Integrity**: Always ensure claims data used for features occurred BEFORE the authorization submission_date
- **Business Constraints**: Balance false positive cost (incorrect approvals) vs false negative cost (missed approvals requiring manual review)

### Important Considerations
- False negatives are costly: missed auto-approvals mean unnecessary manual review and delays
- False positives are risky: incorrect auto-approvals can lead to fraud or financial loss
- The project aims to reduce manual review workload while maintaining acceptable precision

## Testing and Validation

When making changes:
1. Run the notebook cells in sequence to ensure no errors
2. Verify that data transformations maintain temporal integrity
3. Check that model performance metrics are calculated correctly
4. Validate that visualizations render properly and tell the right story
5. Ensure any new features don't introduce data leakage

## Documentation

- Update `PROJECT_PLAN.md` if adding new phases or tasks
- Document any significant model changes or findings in the notebook with markdown cells
- Add entries to `FIXES_SUMMARY.md` for bug fixes or improvements
- Include rationale for modeling decisions (e.g., why a specific threshold was chosen)

## Common Tasks

### Adding a New Feature
1. Ensure temporal validity (only historical data)
2. Create feature in both train and test sets consistently
3. Document the feature's business meaning
4. Check for correlation with existing features
5. Test feature importance in the model

### Tuning a Model
1. Define the evaluation metric aligned with business goals
2. Use cross-validation with temporal awareness
3. Document the hyperparameter search space and results
4. Compare against baseline and previous best model
5. Analyze where the model succeeds and fails (especially false negatives)

### Creating Visualizations
1. Use consistent color schemes (seaborn default palette recommended)
2. Add clear titles, axis labels, and legends
3. Annotate key insights directly on the chart
4. Keep visualizations simple and focused on one message
5. Save high-resolution figures for presentations

## Security and Privacy

- Do not commit actual sensitive data to the repository
- Use sample/anonymized data for examples and testing
- Be mindful of PII (Personally Identifiable Information) in datasets
- Clear notebook outputs if they contain sensitive information

## Collaboration

- Keep notebook cells clean and focused
- Document assumptions and decisions in markdown cells
- Use meaningful commit messages
- If modifying existing analysis, explain why in comments or markdown

## Questions or Issues?

Refer to `PROJECT_PLAN.md` for detailed project phases, objectives, and tasks. The plan includes comprehensive guidance on data cleaning, feature engineering, model development, and business recommendations.
