# Credit Risk Prediction

A machine learning project for predicting loan defaults using loan characteristics, lender information, customer borrowing history, and Kenyan economic indicators.

## Project Overview

This project develops an end-to-end machine learning pipeline for loan default prediction.

The workflow covers:

- Data preparation
- Exploratory data analysis
- Feature engineering
- Model training
- Model evaluation
- Classification threshold tuning
- Model artifact saving
- API deployment using FastAPI

The final model is an XGBoost classifier. The trained model and preprocessing pipeline are saved as artifacts and used by a FastAPI application for prediction.

## Business Problem

The objective is to predict whether a loan will default based on information available about the loan, lender, customer history, and economic conditions.

The target variable is:

- `0` — non-default
- `1` — default

Because the target is highly imbalanced, model evaluation uses precision, recall, F1-score, and ROC-AUC in addition to accuracy.

## Dataset

The primary dataset contains 68,654 lender-level loan records.

The dataset includes:

- Loan characteristics
- Loan and repayment amounts
- Loan duration
- Lender information
- New versus repeat loan status
- Customer borrowing history
- Loan disbursement dates
- Lender funding information
- Economic indicators for Kenya

### Target Distribution

| Outcome | Records | Proportion |
|---|---:|---:|
| Non-default (`0`) | 67,396 | 98.17% |
| Default (`1`) | 1,258 | 1.83% |

### Dataset Structure

The dataset contains 66,569 unique loan IDs across 68,654 records. Some loans appear more than once because they were associated with multiple lenders.

This was considered during feature engineering. Historical customer features were calculated using records with an earlier disbursement date so that information from the same date was not used as prior history.

## Business Questions

The analysis focuses on the following questions:

1. How do default rates vary across loan types?
2. How do default rates differ between new and repeat loans?
3. What patterns are observed across lenders?
4. Does previous customer borrowing history provide useful predictive information?
5. Do loan amounts and repayment characteristics relate to default?
6. Do economic indicators provide additional information for prediction?
7. How do different classification models perform on the imbalanced target?
8. How does the classification threshold affect precision and recall?

## Data Preparation

The data preparation process included:

- Standardizing column names
- Converting date columns to datetime format
- Checking missing values
- Checking duplicate records
- Examining categorical and numerical variables
- Identifying constant columns
- Investigating the relationship between loan IDs and lender records
- Separating features from the target variable

The `country_id` column contained only one country in the primary dataset and therefore did not provide useful variation for modeling.

Identifier columns were excluded from the model while being retained where needed for analysis and feature engineering.

## Exploratory Data Analysis

Exploratory analysis was performed on the target variable, numerical variables, and categorical variables.

### Target Imbalance

Defaults represented approximately 1.83% of the observations. This imbalance was considered during model training and evaluation.

### Loan Type

Default rates varied across loan types. Some smaller loan-type categories had high observed default rates, but their small sample sizes were considered when interpreting the results.

### New vs Repeat Loans

New loans had a higher observed default rate than repeat loans in the dataset.

- New Loan: approximately 19.93%
- Repeat Loan: approximately 1.69%

Repeat loans made up the large majority of observations.

### Lender

Observed default rates varied between lenders. The relationship between lender and loan type was also examined because some lenders were concentrated in particular loan types.

### Numerical Variables

Several numerical variables were strongly correlated:

- `total_amount` and `total_amount_to_repay`
- `amount_funded_by_lender` and `lender_portion_to_be_repaid`

These relationships were retained for the initial modeling stage and can be revisited during further model refinement.

## Feature Engineering

Additional features were created from the available loan and customer information.

### Date Features

The loan disbursement date was used to create:

- `loan_year`
- `loan_month`
- `loan_day_of_week`

### Financial Features

Additional financial features included:

- `interest_amount`
- `repayment_ratio`
- `repayment_ratio_zero`

### Customer History

Historical customer features included:

- `previous_loan_count`
- `previous_default_count`
- `previous_default_rate`
- `days_since_previous_loan`
- `first_loan`

Historical features were calculated using strictly earlier loan records.

### Economic Features

Economic indicators for Kenya were merged using the loan year.

The final economic features included:

- `deposit_interest_rate`
- `inflation_rate`
- `interest_rate_spread`
- `lending_interest_rate`
- `exchange_rate`
- `real_interest_rate`
- `unemployment_rate`
- `economic_data_missing`

The economic dataset contained data through 2023, while some loan records were from 2024. The `economic_data_missing` feature was therefore included to identify records without corresponding economic data.

## Modeling

The data was split into training and validation sets using a stratified split.

Categorical features were one-hot encoded, while numerical features were median-imputed and standardized.

The following models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Class imbalance was addressed using class weighting for the scikit-learn models and `scale_pos_weight` for XGBoost.

## Model Evaluation

The baseline validation results were:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 94.31% | 23.79% | 95.24% | 38.07% | 98.83% |
| Decision Tree | 97.98% | 47.45% | 96.03% | 63.52% | 97.48% |
| Random Forest | 99.42% | 82.40% | 87.30% | 84.78% | 99.41% |
| XGBoost | 99.30% | 75.32% | 92.06% | 82.86% | 99.60% |

Accuracy was not used as the only selection criterion because of the imbalance in the target variable.

## Threshold Tuning

The XGBoost model produces a probability of default. The default classification threshold was initially 0.50.

Different thresholds were evaluated to examine the trade-off between precision and recall.

The selected candidate threshold was **0.89**, which produced:

- Precision: 88.57%
- Recall: 86.11%
- F1-score: 87.32%

At this threshold, the validation confusion matrix was:

```text
[[13451,    28],
 [   35,   217]]
```

The threshold is stored separately in `models/threshold.json` so that the same decision threshold can be used during deployment.

## Final Model

The final model uses XGBoost with:

- `n_estimators = 200`
- `max_depth = 6`
- `learning_rate = 0.1`
- `scale_pos_weight ≈ 53.60`

The model, preprocessing pipeline, and threshold are saved as separate artifacts.

## API Deployment

The model is currently served locally through a FastAPI application.

The API workflow is:

```text
JSON request
    ↓
FastAPI
    ↓
DataFrame
    ↓
Preprocessing pipeline
    ↓
XGBoost model
    ↓
Default probability
    ↓
Classification threshold
    ↓
JSON response
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Checks that the API is running |
| POST | `/predict` | Returns a loan default prediction |

### Example Request

```json
{
  "lender_id": 267278,
  "loan_type": "Type_9",
  "total_amount": 23541.0,
  "total_amount_to_repay": 23895.0,
  "duration": 31,
  "new_versus_repeat": "Repeat Loan",
  "amount_funded_by_lender": 1540.5,
  "lender_portion_funded": 0.0654390212820186,
  "lender_portion_to_be_repaid": 1564.0,
  "loan_year": 2023,
  "loan_month": 7,
  "loan_day_of_week": 5,
  "interest_amount": 354.0,
  "repayment_ratio": 1.0150375939849623,
  "repayment_ratio_zero": 0,
  "previous_loan_count": 49,
  "previous_default_count": 0,
  "previous_default_rate": 0.0,
  "days_since_previous_loan": 14.0,
  "first_loan": 0,
  "deposit_interest_rate": 9.16769017629068,
  "inflation_rate": 7.67139634029402,
  "interest_rate_spread": 4.42081153983732,
  "lending_interest_rate": 13.588501716128,
  "exchange_rate": 139.846383759617,
  "real_interest_rate": 6.54651706101945,
  "unemployment_rate": 5.682,
  "economic_data_missing": 0
}
```

### Example Response

```json
{
  "default_probability": 0.0008863278781063855,
  "threshold": 0.89,
  "prediction": 0,
  "result": "non-default"
}
```
## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- FastAPI
- Uvicorn
- Joblib
- uv

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/PravinMaleya/Credit-Risk-Prediction.git
cd credit-risk-prediction
```

### 2. Install dependencies

This project uses `uv` for Python environment and dependency management.

```bash
uv sync
```

### 3. Run the API

```bash
uv run uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI's interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### 4. Make a prediction

Use the interactive documentation at `/docs` to send a `POST` request to `/predict` with the required feature values.

## Key Findings

- The target variable is highly imbalanced, with defaults representing approximately 1.83% of observations.
- Default rates vary considerably across loan types.
- New loans have a higher observed default rate than repeat loans in this dataset.
- Lender-level default rates vary, with loan-type composition also differing between lenders.
- Previous customer borrowing history provides additional features for the model.
- Economic indicators can be incorporated based on the loan year, although economic data was unavailable for some 2024 records.
- XGBoost produced a high ROC-AUC and provided a useful balance between precision and recall after threshold tuning.

## Limitations

- The dataset is highly imbalanced.
- Economic indicators are annual values and may not represent the exact information available at the time each loan was issued.
- The API currently expects the engineered model features rather than calculating all feature engineering from raw loan information.
- The model was evaluated using a validation split from the available dataset and has not been evaluated on an independent production dataset.
- The classification threshold was selected using validation data and should be reassessed with additional validation or business cost information in a production setting.

## Future Improvements

Potential improvements include:

- Evaluating the model on a separate holdout dataset
- Testing additional models and hyperparameters
- Calibrating predicted probabilities
- Evaluating business-specific costs of false positives and false negatives
- Moving feature engineering into the production pipeline
- Adding automated tests for the API
- Containerizing and deploying the API
