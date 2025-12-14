# Telco Customer Churn Prediction

## Project Overview

This project analyzes and predicts customer churn in the telecommunications industry using machine learning techniques. The objective is to identify key factors influencing customer retention and attrition, providing actionable insights to improve customer loyalty and reduce churn rates.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Methodology](#methodology)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Feature Engineering](#feature-engineering)
  - [Statistical Testing](#statistical-testing)
  - [Machine Learning Models](#machine-learning-models)
- [Key Findings](#key-findings)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Usage](#usage)
- [Contributors](#contributors)

## Dataset

**Source:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains customer information from a telecommunications company, including:
- Customer demographics (gender, senior citizen status, partner, dependents)
- Account information (tenure, contract type, payment method, billing preferences)
- Service subscriptions (phone service, internet service, streaming services, online security)
- Churn status (target variable)

## Installation

### Prerequisites

Ensure you have R installed (version 3.6 or higher recommended). Install the required packages:

```r
install.packages(c("dplyr", "data.table", "ISLR", "caret", "ggplot2",
                   "epiDisplay", "pROC", "randomForest", "xgboost", "mxnet"))
```

### Running the Analysis

1. Clone this repository
2. Download the dataset from the source link above and place it in the project directory
3. Open [Group5_final_project.R](Group5_final_project.R) in RStudio or your preferred R environment
4. Run the script sequentially

## Methodology

### Exploratory Data Analysis

Comprehensive univariate and bivariate analyses were performed to understand:
- Customer demographics distribution
- Churn rate patterns
- Tenure distribution by year ranges
- Contract type preferences
- Service usage patterns

Key visualizations include:
- Pie charts for categorical variables (gender, churn rate, contract types)
- Bar plots for tenure distribution
- Comparative analysis of senior citizens vs. churn rates

### Feature Engineering

- **Data Preprocessing:** Missing values and infinite values were handled appropriately
- **Feature Scaling:** Centering and scaling applied using `preProcess()` function
- **Train-Test Split:** 70-30 split with random seed for reproducibility
- **Class Imbalance Handling:**
  - Upsampling technique applied
  - Downsampling technique applied

### Statistical Testing

Chi-square tests were conducted to identify significant relationships between categorical variables and churn:
- Partner status: χ² = 157.5, p < 0.001
- Dependents: χ² = 186.32, p < 0.001
- Streaming services: χ² > 370, p < 0.001
- Payment method: χ² = 645.43, p < 0.001
- Gender and phone service showed no significant relationship with churn

ANOVA tests confirmed:
- Tenure significantly affects churn (p < 0.001)
- Total charges significantly affect churn (p < 0.001)

### Machine Learning Models

Four classification models were implemented and evaluated using 10-fold cross-validation:

1. **Logistic Regression**
   - Forward stepwise feature selection
   - Adjusted odds ratios calculated
   - ROC curve analysis with optimal threshold selection

2. **Random Forest**
   - Variable importance analysis
   - Hyperparameter tuning (mtry = 1-10)
   - ntree = 100, nodesize = 75

3. **XGBoost**
   - Grid search for optimal parameters
   - Parameters: max_depth, eta, colsample_bytree, subsample, min_child_weight

4. **Neural Network (MXNet)**
   - Three-layer architecture (4-2-0 nodes)
   - ReLU activation function
   - Dropout regularization (0.1)

All models were evaluated on:
- Confusion matrix metrics (accuracy, sensitivity, specificity)
- ROC-AUC scores
- Training and testing performance

## Key Findings

Based on variable importance analysis and statistical testing:

**Top Predictive Features:**
1. Contract type (month-to-month contracts have highest churn)
2. Tenure (shorter tenure correlates with higher churn)
3. Internet service type
4. Payment method
5. Total charges and monthly charges
6. Additional services (online security, tech support, streaming services)

**Customer Segments at Risk:**
- Month-to-month contract holders
- New customers (0-1 year tenure)
- Customers without online security or tech support
- Electronic check payment users
- Customers with paperless billing

## Results

Model performance comparison on test set:
- Each model was evaluated using confusion matrices and ROC curves
- Performance metrics include accuracy, sensitivity, specificity, and AUC
- ROC curves with optimal threshold selection using Youden's index

*(Note: Specific performance metrics can be obtained by running the analysis script)*

## Technologies Used

- **Language:** R (version 3.6+)
- **Libraries:**
  - Data manipulation: `dplyr`, `data.table`
  - Visualization: `ggplot2`
  - Machine learning: `caret`, `randomForest`, `xgboost`, `mxnet`
  - Statistical analysis: `epiDisplay`, `pROC`, `ISLR`

## Usage

To reproduce the analysis:

```r
# Load the main script
source("Group5_final_project.R")

# The script will automatically:
# 1. Load and preprocess the data
# 2. Perform exploratory data analysis
# 3. Conduct statistical tests
# 4. Train and evaluate all models
# 5. Generate visualizations and performance metrics
```

## Contributors

Group 5 - Introduction to Big Data Analytics Course
