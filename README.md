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
| Start Date | December 1, 2009 |
| End Date | December 9, 2011 |

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
6. Temporal Churn Dataset Construction
7. Model Training and Evaluation
8. Explainable AI Using SHAP
9. Business Recommendations
10. Model and Report Saving

## Data Preprocessing

The following preprocessing operations were performed:

- Removed transactions with missing customer identifiers.
- Removed cancelled invoices.
- Excluded transactions with non-positive quantities and prices.
- Removed exact duplicate records.
- Converted invoice dates into datetime format.
- Converted customer identifiers into categorical identifiers.
- Created a TotalPrice feature by multiplying Quantity and Price.

After preprocessing, 779,425 transaction records were retained, with no missing values or duplicate records.

Returns and cancellations were excluded because the project focuses on positive purchasing behavior. Consequently, reported sales represent retained purchase transactions rather than net revenue.

## Exploratory Data Analysis

Key findings from exploratory data analysis include:

- Sales exhibit seasonal patterns, with notable peaks around October and November.
- The United Kingdom accounts for the largest share of sales.
- Customer spending and purchase frequency exhibit right-skewed distributions.
- The median customer placed three orders.
- Average customer spending was approximately £2,955.90, compared with median spending of £867.74.
- A substantial proportion of customers made repeat purchases.

The analysis highlights differences in customer purchasing behavior and motivates the application of customer segmentation and churn prediction.

## RFM Feature Engineering

Customer-level features were constructed using RFM analysis.

| Feature | Description |
|---|---|
| Recency | Number of days since the customer's last purchase |
| Frequency | Number of unique purchase invoices |
| Monetary | Total customer spending |

Logarithmic transformation was applied to reduce the influence of highly skewed feature distributions. StandardScaler was subsequently used to balance feature scales before clustering.

The original RFM values were retained for customer segment interpretation.

## Customer Segmentation

Two unsupervised learning algorithms were implemented:

- K-Means Clustering
- Agglomerative Hierarchical Clustering

The Elbow Method and Silhouette Score were used to evaluate different cluster configurations. Principal Component Analysis (PCA) was applied for two-dimensional visualization of the resulting customer segments.

### Clustering Performance

| Algorithm | Number of Clusters | Silhouette Score |
|---|---:|---:|
| K-Means | 4 | **0.3664** |
| Hierarchical Clustering | 4 | 0.3179 |

Although K-Means with two clusters achieved the highest Silhouette Score of 0.4385, four clusters were selected to provide more granular and actionable business segmentation.

This selection represents a trade-off between mathematical cluster separation and practical business interpretability.

### Identified Customer Segments

| Customer Segment | Customers | Avg. Recency (Days) | Avg. Frequency | Avg. Spending |
|---|---:|---:|---:|---:|
| High-Value Loyal | 1,204 | 28.44 | 19.21 | £10,678.40 |
| At-Risk | 1,455 | 231.20 | 5.03 | £1,945.08 |
| Active Occasional | 1,261 | 29.24 | 3.03 | £847.75 |
| Inactive Low-Value | 1,958 | 397.89 | 1.38 | £316.09 |

### Customer and Sales Distribution

| Customer Segment | Customer Share | Sales Contribution |
|---|---:|---:|
| High-Value Loyal | 20.48% | 74.00% |
| At-Risk | 24.75% | 16.29% |
| Active Occasional | 21.45% | 6.15% |
| Inactive Low-Value | 33.31% | 3.56% |

### Key Business Insight

Approximately **20.5% of customers contributed 74% of total retained sales**, highlighting the importance of retaining high-value customers.

The moderate Silhouette Score indicates that the identified segments should be interpreted as practical behavioral partitions rather than completely distinct natural customer groups.

## Customer Churn Prediction

Customer churn was operationally defined as the absence of any purchase within a subsequent 90-day period.

| Target Label | Description |
|---|---|
| 0 — Retained | Customer makes at least one purchase within the next 90 days |
| 1 — Churned | Customer makes no purchases within the next 90 days |

Since the original dataset does not contain churn labels, the target variable was constructed using historical and subsequent transaction activity.

### Temporal Train-Test Splitting Strategy

Instead of using a conventional random 80:20 split, this project employs an expanding-window chronological splitting strategy to simulate a real-world customer churn prediction scenario.

Customer-level RFM features are calculated using historical transactions available before each cutoff date, while churn labels are generated from purchasing activity during the subsequent 90-day prediction period.

| Dataset | Cutoff Date | Customer Snapshots | Features |
|---|---|---:|---:|
| Training | March 1, 2011 | 4,537 | 3 |
| Testing | September 1, 2011 | 5,249 | 3 |

### Observation and Prediction Windows

| Dataset | Historical Observation Period | Churn Prediction Period |
|---|---|---|
| Training | Dec 1, 2009 – Feb 28, 2011 | Mar 1 – May 29, 2011 |
| Testing | Dec 1, 2009 – Aug 31, 2011 | Sep 1 – Nov 29, 2011 |

### Why Is the Test Dataset Larger Than the Training Dataset?

The testing dataset is larger because the historical observation window expands over time, incorporating additional customers who made their first recorded purchases between the two cutoff dates.

The training and testing datasets are temporal customer snapshots rather than mutually exclusive customer groups.

Based on the expanding-window construction:

- The training dataset contains 4,537 historical customers.
- The testing dataset contains those historical customers and 712 additional customers.
- The combined 9,786 observations represent customer snapshots at different points in time, not 9,786 unique individuals.

Customer overlap is therefore intentional and reflects a realistic business scenario in which a retailer periodically reassesses the churn risk of its existing customer base.

### Data Leakage Prevention

The following measures were implemented to prevent temporal data leakage:

- RFM features are calculated exclusively from transactions occurring before the respective cutoff date.
- Churn labels are generated from the subsequent 90-day prediction window.
- Training and testing label windows do not overlap.
- Training labels are fully observable before the testing cutoff date.
- Customer identifiers are excluded from model input features.
- Feature scaling is fitted exclusively on the training dataset.
- Full-history RFM features used for descriptive customer segmentation are not reused for churn prediction.

**Evaluation Scope:** This strategy evaluates future churn prediction across an existing and expanding customer base rather than generalization to entirely unseen customers. Since the same customers can appear in both temporal snapshots, the observations are not statistically independent.

### Class Distribution

| Class | Training | Testing |
|---|---:|---:|
| Retained | 1,587 | 2,241 |
| Churned | 2,950 | 3,008 |
| Total | 4,537 | 5,249 |
| Churn Rate | 65.02% | 57.31% |

The datasets exhibit moderate class imbalance. No synthetic oversampling techniques were applied.

### Classification Models

Three classical machine learning algorithms were implemented:

- Logistic Regression
- Random Forest
- XGBoost

Log-transformed and standardized RFM features were used for Logistic Regression, while the original RFM features were used for the tree-based models.

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7148 | **0.6969** | 0.8890 | 0.7813 | 0.7853 | 0.8170 |
| Random Forest | **0.7190** | 0.6963 | **0.9039** | **0.7866** | 0.7755 | 0.8020 |
| XGBoost | 0.7175 | 0.6953 | 0.9026 | 0.7855 | **0.7902** | **0.8186** |

### Majority-Class Baseline Comparison

An analytical majority-class baseline was calculated by assuming that every test customer is predicted as churned.

| Metric | Majority-Class Baseline | XGBoost |
|---|---:|---:|
| Accuracy | 57.31% | **71.75%** |
| Precision | 57.31% | **69.53%** |
| Recall | **100.00%** | 90.26% |
| F1-Score | 0.7286 | **0.7855** |
| ROC-AUC | 0.5000 | **0.7902** |
| Average Precision | 0.5731 | **0.8186** |

The comparison demonstrates that XGBoost provides better discrimination between retained and churned customers than a constant-score majority-class baseline.

However, the high recall should be interpreted alongside precision, class prevalence and false-positive predictions.

### Model Selection

XGBoost was selected as the preferred model for churn risk ranking because it achieved the highest ROC-AUC and Average Precision in the conducted experiments.

Random Forest achieved marginally higher accuracy, recall and F1-score at the default classification threshold.

All three models demonstrated broadly comparable performance. The observed differences are small and have not been established as statistically significant. Therefore, XGBoost is selected for its observed ranking performance rather than a claim of definitive superiority.

### XGBoost Performance Highlights

- Accuracy: **71.75%**
- Recall: **90.26%**
- F1-Score: **78.55%**
- ROC-AUC: **0.7902**
- Average Precision: **0.8186**

The model successfully identified 2,715 out of 3,008 customers who made no purchases during the prediction window.

### Confusion Matrix

| Actual / Predicted | Retained | Churned |
|---|---:|---:|
| Retained | 1,051 | 1,190 |
| Churned | 293 | 2,715 |

The model demonstrates high sensitivity to customer churn but generates a considerable number of false-positive predictions. This creates a potential trade-off between identifying at-risk customers and allocating unnecessary retention resources.

## Explainable AI Using SHAP

SHAP was used to interpret the XGBoost model through global feature importance, beeswarm plots and individual customer-level explanations.

### Global Feature Importance

| Feature | Mean Absolute SHAP Value |
|---|---:|
| Recency | 0.66 |
| Frequency | 0.57 |
| Monetary | 0.29 |

The analysis revealed that:

- Higher Recency generally increases predicted churn risk.
- Higher Frequency generally reduces predicted churn risk.
- Higher Monetary values generally reduce predicted churn risk.

Recency was identified as the most influential feature in the model's churn predictions, followed by Frequency and Monetary.

These findings are consistent with intuitive purchasing behavior and demonstrate that the model has learned interpretable relationships from the RFM features.

SHAP values explain model predictions rather than establishing causal relationships between customer behavior and churn.

## Business Recommendations

| Customer Segment | Recommended Strategy |
|---|---|
| High-Value Loyal | Loyalty rewards and exclusive offers |
| At-Risk | Personalized reactivation campaigns |
| Active Occasional | Cross-selling and repeat-purchase incentives |
| Inactive Low-Value | Low-cost win-back campaigns |

Customer churn scores were categorized into illustrative risk groups to demonstrate how model predictions could support targeted retention strategies.

| Risk Category | Estimated Churn Score |
|---|---|
| Low Risk | Below 0.50 |
| Medium Risk | 0.50 to below 0.80 |
| High Risk | 0.80 and above |

These thresholds are illustrative and have not been optimized using business costs or campaign outcomes. The model's probability estimates have not been independently calibrated.

The proposed recommendations are potential business strategies rather than experimentally verified retention outcomes.

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

1. Clone the repository.
2. Download the [Online Retail II dataset](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci).
3. Open the project notebook in Kaggle, Jupyter Notebook or Google Colab.
4. Install the required Python libraries if necessary.
5. Update the dataset path according to the execution environment.
6. Run the notebook cells sequentially.

The project was developed using Kaggle's CPU environment and does not require a GPU or paid API.

### Model Inference Notes

- The saved K-Means model expects log-transformed and standardized RFM features in the original feature order.
- The saved RFM scaler must be used when transforming new customer data for segmentation.
- The XGBoost model expects the original, unscaled Recency, Frequency and Monetary features.
- The customer segment mapping converts numerical cluster labels into interpretable business categories.

## Limitations

- Churn represents 90-day purchasing inactivity rather than confirmed permanent customer loss.
- Customer overlap across temporal snapshots means that evaluation does not measure generalization to entirely unseen customers.
- Expanding historical observation windows provide different amounts of purchasing history at the training and testing cutoff dates.
- Only three RFM features were used for churn prediction, limiting the behavioral information available to the models.
- The models were compared using a single future test period, which was also used to select the preferred model.
- No confidence intervals or statistical significance tests were conducted to establish differences between classification models.
- A recency-only predictive baseline was not evaluated, so the incremental contribution of the remaining features has not been isolated.
- The clustering models exhibit moderate rather than strong cluster separation.
- The churn prediction model generates a considerable number of false positives.
- Estimated churn probabilities have not been independently calibrated.
- Sales analysis excludes returns, cancellations and transactions without customer identifiers.

## Future Improvements

Potential extensions of this project include:

- Incorporating additional behavioral features such as customer tenure, average order value, purchase intervals and product diversity.
- Evaluating a recency-only baseline to measure the incremental predictive value of the full RFM feature set.
- Introducing fixed-length observation windows and multiple chronological validation periods.
- Estimating confidence intervals for model performance.
- Calibrating churn probabilities using Platt scaling or isotonic regression.
- Optimizing classification thresholds based on customer retention costs and expected business benefits.

## Conclusion

This project demonstrates the practical application of classical machine learning to e-commerce customer intelligence.

Through RFM-based segmentation, chronological churn prediction and SHAP explainability, the system identifies meaningful customer groups and provides interpretable estimates of future purchasing inactivity.

The results demonstrate how supervised and unsupervised learning can support customer retention planning, targeted marketing and data-driven business decision-making, while acknowledging the methodological limitations and opportunities for further improvement.
