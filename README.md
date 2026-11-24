# E-Commerce Customer Intelligence Using Machine Learning

## Project Overview

An end-to-end classical machine learning project focused on understanding customer purchasing behavior, identifying valuable customer segments and predicting customer churn.

The project combines RFM analysis, unsupervised learning, supervised learning and Explainable AI (XAI) to generate actionable business insights for customer retention and targeted marketing.

## Project Objectives

- Analyze customer purchasing behavior and sales trends.
- Segment customers based on Recency, Frequency and Monetary (RFM) features.
- Predict customer churn using classical machine learning algorithms.
- Compare model performance using appropriate evaluation metrics.
- Explain model predictions using SHAP.
- Generate data-driven customer retention recommendations.

## Dataset

**Online Retail II — UCI**

The dataset contains real-world transactions from a UK-based online retailer between December 2009 and December 2011.

[Dataset Source — Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)

| Attribute | Value |
|---|---:|
| Original Records | 1,067,371 |
| Cleaned Records | 779,425 |
| Unique Customers | 5,878 |
| Unique Products | 4,631 |
| Total Orders | 36,969 |
| Total Retained Sales | £17,374,804.27 |
| Average Order Value | £469.98 |

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Joblib
- Kaggle Notebooks

## Project Workflow

1. Data Loading and Understanding
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. RFM Feature Engineering
5. Customer Segmentation
6. Customer Churn Prediction
7. Model Training and Evaluation
8. Explainable AI Using SHAP
9. Business Recommendations
10. Model and Report Saving

## Exploratory Data Analysis

Key findings from the exploratory data analysis include:

- Sales exhibit seasonal patterns, with notable peaks around October and November.
- The United Kingdom accounts for the largest share of sales.
- Customer spending and purchase frequency exhibit right-skewed distributions.
- 72.4% of customers are repeat customers, while 27.6% are one-time customers.

## RFM Feature Engineering

Customer-level features were constructed using RFM analysis.

| Feature | Description |
|---|---|
| Recency | Number of days since the customer's last purchase |
| Frequency | Number of unique purchase invoices |
| Monetary | Total customer spending |

Logarithmic transformation and standardization were applied before clustering to reduce skewness and balance feature scales.

## Customer Segmentation

Two unsupervised learning algorithms were implemented:

- K-Means Clustering
- Agglomerative Hierarchical Clustering

The Elbow Method and Silhouette Score were used to evaluate different cluster configurations. PCA was applied for two-dimensional visualization.

### Clustering Performance

| Algorithm | Number of Clusters | Silhouette Score |
|---|---:|---:|
| K-Means | 4 | **0.3664** |
| Hierarchical Clustering | 4 | 0.3179 |

Although two clusters achieved a higher Silhouette Score, four clusters were selected to provide more meaningful and actionable business segmentation.

### Identified Customer Segments

| Customer Segment | Customers | Avg. Recency | Avg. Frequency | Avg. Spending |
|---|---:|---:|---:|---:|
| High-Value Loyal | 1,204 | 28.44 | 19.21 | £10,678.40 |
| At-Risk | 1,455 | 231.20 | 5.03 | £1,945.08 |
| Active Occasional | 1,261 | 29.24 | 3.03 | £847.75 |
| Inactive Low-Value | 1,958 | 397.89 | 1.38 | £316.09 |

### Key Business Insight

Approximately **20.5% of customers contributed 74% of total retained sales**, highlighting the importance of retaining high-value customers.

## Customer Churn Prediction

Customer churn was defined as the absence of any purchase within a subsequent 90-day period.

Historical RFM features and future purchase behavior were separated using chronological observation and prediction windows to prevent temporal data leakage.

| Dataset | Customers | Features |
|---|---:|---:|
| Training | 4,537 | 3 |
| Testing | 5,249 | 3 |

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7148 | **0.6969** | 0.8890 | 0.7813 | 0.7853 | 0.8170 |
| Random Forest | **0.7190** | 0.6963 | **0.9039** | **0.7866** | 0.7755 | 0.8020 |
| XGBoost | 0.7175 | 0.6953 | 0.9026 | 0.7855 | **0.7902** | **0.8186** |

XGBoost was selected as the preferred churn prediction model due to its highest ROC-AUC and Average Precision, while maintaining competitive recall and F1-score.

### XGBoost Performance Highlights

- Accuracy: **71.75%**
- Recall: **90.26%**
- F1-Score: **78.55%**
- ROC-AUC: **0.7902**
- Average Precision: **0.8186**

The model successfully identified 2,715 out of 3,008 customers who made no purchases during the prediction window.

## Explainable AI (SHAP)

SHAP was used to interpret the XGBoost model through global feature importance, beeswarm plots and individual customer-level explanations.

| Feature | Mean Absolute SHAP Value |
|---|---:|
| Recency | 0.66 |
| Frequency | 0.57 |
| Monetary | 0.29 |

The analysis revealed that:

- Higher Recency generally increases predicted churn risk.
- Higher Frequency generally reduces predicted churn risk.
- Higher Monetary values generally reduce predicted churn risk.

Recency was identified as the most influential feature in the model's churn predictions.

## Business Recommendations

| Customer Segment | Recommended Strategy |
|---|---|
| High-Value Loyal | Loyalty rewards and exclusive offers |
| At-Risk | Personalized reactivation campaigns |
| Active Occasional | Cross-selling and repeat-purchase incentives |
| Inactive Low-Value | Low-cost win-back campaigns |

Customer churn probabilities were also categorized into low, medium and high-risk groups using illustrative thresholds to support targeted retention strategies.

## Saved Models and Outputs

| File | Description |
|---|---|
| `kmeans_model.pkl` | Trained customer segmentation model |
| `rfm_scaler.pkl` | Fitted RFM feature scaler |
| `segment_names.pkl` | Customer segment label mapping |
| `xgboost_churn_model.json` | Trained churn prediction model |
| `customer_segments.csv` | Customer segmentation results |
| `churn_predictions.csv` | Historical churn predictions and risk levels |
| `model_comparison.csv` | Classification model evaluation results |

## How to Run

1. Clone this repository.
2. Download the Online Retail II dataset from Kaggle.
3. Open the Jupyter Notebook in Kaggle, Jupyter Notebook or Google Colab.
4. Install the required Python libraries if necessary.
5. Update the dataset path according to your environment.
6. Run the notebook cells sequentially.

The project was developed using Kaggle's CPU environment and does not require a GPU or paid API.

## Limitations

- Churn represents 90-day purchasing inactivity rather than confirmed permanent customer loss.
- The clustering models exhibit moderate cluster separation.
- The classification models were compared using a single future test period.
- The churn prediction model produces a considerable number of false positives.
- Predicted probabilities have not been independently calibrated.
- Sales analysis excludes returns, cancellations and transactions without customer identifiers.

## Conclusion

This project demonstrates the practical application of classical machine learning to e-commerce customer intelligence. By combining customer segmentation, churn prediction and explainable AI, the system provides meaningful insights into customer behavior and supports data-driven retention and marketing strategies.
