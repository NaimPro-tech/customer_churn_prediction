## About Dataset

Customer churn is a major issue in telecommunications, where retaining customers is significantly more cost-effective than acquiring new ones. This dataset contains customer-level details including demographics, services subscribed, billing information, and whether the customer churned.

## Dataset Size:

Rows: 7043
Columns: 21

Columns name and details:

CustomerID: Unique identifier

Gender: Male/Female

SeniorCitizen: 0 = No, 1 = Yes

Partner: Yes/No

Dependents: Yes/No

Tenure: Number of months with the company

PhoneService: Yes/No

MultipleLines: Yes/No/No phone service

InternetService: DSL/Fiber optic/No

OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies: Yes/No/No internet service

Contract: Month-to-month/One year/Two year

PaperlessBilling: Yes/No

PaymentMethod: Electronic check/Mailed check/Bank transfer/Credit card

MonthlyCharges: Amount billed monthly

TotalCharges: Lifetime total charges

Churn: Yes/No

### Potential Uses:

### Predictive Modeling: Build machine learning models to forecast churn.
### Exploratory Data Analysis: Visualize customer behavior patterns.
### Business Strategy: Identify high-risk customer segments for retention.


## Why not Neural Networks?
- Dataset size is moderate
- No spatial or sequential structure
- Tree-based boosting models outperform ANN on tabular data
- Better interpretability and deployment simplicity


## Acknowledgements

This dataset is adapted from the IBM Sample Data Sets and is intended for educational and research use only.

