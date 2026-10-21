# German Credit Risk Prediction using Random Forest

A machine learning project for predicting **credit risk** using the German Credit dataset. The project implements a complete preprocessing, feature engineering, model training, hyperparameter optimization, and ROC-AUC evaluation pipeline using a **Random Forest Classifier**.

The objective is to classify applicants into:

- **Good Credit Risk**
- **Bad Credit Risk**

---

## Project Overview

Credit risk prediction is a binary classification problem where the goal is to determine whether a loan applicant represents a good or bad credit risk based on demographic, financial, and credit-related information.

This project develops a Random Forest-based classification pipeline that includes:

- Data preprocessing
- Missing-value imputation
- Categorical variable encoding
- Feature engineering
- Feature scaling
- Stratified train-test splitting
- Random Forest classification
- Randomized hyperparameter search
- 5-fold stratified cross-validation
- ROC-AUC evaluation
- ROC curve visualization

---

## Dataset

The project uses the **German Credit dataset**, containing information about 1,000 credit applicants.

### Dataset Statistics

| Property | Value |
|---|---:|
| Number of samples | 1,000 |
| Target variable | `Risk` |
| Target classes | `good`, `bad` |
| Good-risk samples | 700 |
| Bad-risk samples | 300 |
| Test set | 20% |
| Training set | 80% |

The original dataset contains an unnecessary index column (`Unnamed: 0`), which is removed during preprocessing.

---

## Features

The original input features used by the model are:

| Feature | Description |
|---|---|
| `Age` | Age of the applicant |
| `Sex` | Sex of the applicant |
| `Job` | Job classification |
| `Housing` | Housing status |
| `Saving accounts` | Savings account category |
| `Checking account` | Checking account category |
| `Credit amount` | Amount of credit requested |
| `Duration` | Duration of the credit |
| `Purpose` | Purpose of the credit |

The target variable is:

```text
Risk
```

with the following encoding:

```text
good → 0
bad  → 1
```

Therefore, the model's predicted probability for class `1` represents the estimated probability of **bad credit risk**.

---

## Data Preprocessing

The dataset is first separated into input features and the target variable.

A **stratified 80/20 train-test split** is used:

```text
Training set → 80%
Testing set  → 20%
```

Stratification ensures that the relative proportion of good and bad credit-risk samples is maintained between the training and testing sets.

---

## Feature Engineering

Three numerical features are derived from the original variables.

### Credit per Month

```text
Credit_per_month = Credit amount / Duration
```

This represents the approximate credit amount relative to the duration of the loan.

### Credit-to-Age Ratio

```text
Credit_age_ratio = Credit amount / Age
```

This relates the requested credit amount to the applicant's age.

### Duration-to-Age Ratio

```text
Duration_age_ratio = Duration / Age
```

This provides a normalized representation of loan duration relative to applicant age.

---

## Binned Features

Additional categorical features are created using age, credit amount, and loan duration.

### Age Groups

```text
young
adult
middle_age
senior
```

### Duration Groups

```text
short
medium
long
very_long
```

### Credit Groups

```text
low
medium
high
very_high
```

These grouped variables allow the model to capture potentially nonlinear relationships between these variables and credit risk.

---

## Missing Value Handling

Missing values are handled separately for numerical and categorical features.

### Numerical Features

Missing numerical values are replaced using the **median**:

```python
SimpleImputer(strategy="median")
```

### Categorical Features

Missing categorical values are replaced using the **most frequent value**:

```python
SimpleImputer(strategy="most_frequent")
```

This preprocessing is implemented using a Scikit-learn `ColumnTransformer` and `Pipeline`.

---

## Categorical Encoding

Categorical variables are converted into numerical representations using **One-Hot Encoding**.

The encoder uses:

```python
OneHotEncoder(handle_unknown="ignore")
```

Using `handle_unknown="ignore"` ensures that previously unseen categories in the test data do not cause the preprocessing pipeline to fail.

---

## Feature Scaling

Numerical features are standardized using:

```python
StandardScaler()
```

The preprocessing operations are contained within the machine learning pipeline to ensure that transformations are consistently applied during training and prediction.

---

## Machine Learning Model

### Random Forest Classifier

The primary machine learning algorithm used in this project is the **Random Forest Classifier**.

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

The model is suitable for this problem because it can capture nonlinear relationships and interactions between applicant characteristics without requiring explicit mathematical assumptions about the relationship between the features and credit risk.

---

## Hyperparameter Optimization

Instead of relying on the default Random Forest configuration, the project performs hyperparameter optimization using:

```python
RandomizedSearchCV
```

A total of **60 random hyperparameter combinations** are evaluated.

The search uses:

```text
5-fold Stratified Cross-Validation
```

and the optimization objective is:

```text
ROC-AUC
```

### Hyperparameters Explored

#### Number of Trees

```text
300
500
700
900
1200
```

#### Maximum Tree Depth

```text
3
4
5
6
8
10
12
15
20
None
```

#### Minimum Samples for Splitting

```text
2
5
10
15
20
30
```

#### Minimum Samples per Leaf

```text
1
2
4
6
8
10
15
```

#### Maximum Features

```text
sqrt
log2
None
```

#### Split Criterion

```text
gini
entropy
```

#### Class Weight

```text
None
balanced
balanced_subsample
```

#### Bootstrap

```text
True
```

---

## Cross-Validation

A **5-fold Stratified Cross-Validation** strategy is used during hyperparameter optimization.

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

Stratification ensures that each fold maintains a similar distribution of good and bad credit-risk samples.

---

## Model Evaluation

The final model is evaluated on the held-out test set.

The primary evaluation metric is:

### ROC-AUC

The **Receiver Operating Characteristic – Area Under the Curve (ROC-AUC)** is used as the main evaluation metric.

ROC-AUC evaluates how effectively the model separates:

```text
Good Credit Risk
        vs.
Bad Credit Risk
```

across different classification thresholds.

The predicted probability of the bad-risk class is obtained using:

```python
rf_search.predict_proba(X_test)[:, 1]
```

The resulting probability scores are used to calculate the test ROC-AUC.

---

## ROC Curve

The project generates a ROC curve using:

- False Positive Rate
- True Positive Rate

The diagonal line represents random classification performance.

The trained Random Forest model is plotted against this baseline, with the corresponding ROC-AUC displayed in the legend.

---

## Results

The optimized Random Forest model achieved the following performance on the held-out test set:

| Metric | Score |
|---|---:|
| **ROC-AUC** | **0.7110** |

The ROC curve below shows the model's ability to distinguish between good and bad credit-risk applicants across different classification thresholds.

### ROC Curve

The model achieved an **ROC-AUC of 0.7110**, indicating that the trained Random Forest performs better than random classification in distinguishing between the two credit-risk classes.

The ROC-AUC was calculated using the predicted probability of the **bad-risk class**:

```python
y_prob = rf_search.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, y_prob)
```

The final ROC curve was generated from the false-positive rate and true-positive rate:

```python
fpr, tpr, _ = roc_curve(y_test, y_prob)
```

The diagonal dashed line represents random classification performance, corresponding to an AUC of 0.50.

### Final Performance

```text
Random Forest ROC-AUC: 0.7110
```

The result demonstrates meaningful predictive signal in the selected features, while also indicating that there is considerable room for improvement before the model could be considered suitable for a production credit-risk application.

## Machine Learning Pipeline

The complete workflow is:

```text
German Credit Dataset
          │
          ▼
    Remove Index
          │
          ▼
   Encode Target
   good → 0
   bad  → 1
          │
          ▼
 Stratified Train/Test Split
          │
          ▼
   Feature Engineering
          │
          ├── Credit_per_month
          ├── Credit_age_ratio
          ├── Duration_age_ratio
          ├── Age_group
          ├── Duration_group
          └── Credit_group
          │
          ▼
   Missing Value Imputation
          │
          ▼
    One-Hot Encoding
          │
          ▼
     Standard Scaling
          │
          ▼
    Random Forest Model
          │
          ▼
 Randomized Hyperparameter Search
          │
          ▼
 5-Fold Stratified Cross-Validation
          │
          ▼
      Best Model
          │
          ▼
      Test Set
          │
          ▼
       ROC-AUC
          │
          ▼
       ROC Curve
```

---

## Project Structure

```text
german-credit-risk-ml/
│
├── German_Credit_Risk.ipynb
├── german_credit_data.csv
├── README.md
├── requirements.txt
└── .gitignore
```

### File Description

| File | Description |
|---|---|
| `German_Credit_Risk.ipynb` | Complete machine learning notebook |
| `german_credit_data.csv` | German Credit dataset |
| `README.md` | Project documentation |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files excluded from version control |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/german-credit-risk-ml.git
```

Navigate into the project directory:

```bash
cd german-credit-risk-ml
```

Create a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

The project uses the following Python libraries:

```text
numpy
pandas
scikit-learn
imbalanced-learn
matplotlib
seaborn
scipy
```

Note: `imbalanced-learn` is installed in the notebook, but the current implementation does **not** apply SMOTE, SMOTEENN, or another resampling technique to the training data.

---

## Running the Project

The project is implemented as a Jupyter Notebook and can be executed using either Jupyter Notebook or Google Colab.

### Using Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
German_Credit_Risk.ipynb
```

Ensure that the dataset is in the same directory as the notebook:

```text
German_Credit_Risk.ipynb
german_credit_data.csv
```

The dataset is loaded using:

```python
pd.read_csv("/content/german_credit_data.csv")
```

### Google Colab

The notebook can also be uploaded to Google Colab.

If running outside Google Colab, change the dataset path from:

```python
"/content/german_credit_data.csv"
```

to:

```python
"german_credit_data.csv"
```

when the CSV is stored in the same directory as the notebook.

---

## Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **imbalanced-learn**
- **Matplotlib**
- **Jupyter Notebook**
- **Google Colab**

---

## Key Concepts Demonstrated

This project demonstrates practical implementation of:

- Binary classification
- Credit risk prediction
- Exploratory data preprocessing
- Missing-value imputation
- Feature engineering
- Categorical variable encoding
- Feature scaling
- Stratified train-test splitting
- Random Forest classification
- Hyperparameter optimization
- Randomized Search
- Stratified K-Fold Cross-Validation
- ROC-AUC evaluation
- ROC curve visualization
- Machine learning pipelines

---

## Limitations

The dataset contains only 1,000 observations. Therefore, the results should be interpreted as an experimental evaluation rather than as evidence of production-level credit-risk performance.

Other limitations include:

- Limited dataset size
- Potential dataset-specific patterns
- No external validation dataset
- No probability calibration
- No cost-sensitive evaluation based on actual lending losses
- No fairness or bias analysis
- No production deployment

The ROC-AUC obtained from the test set should therefore not be interpreted as a guarantee of performance on unseen real-world credit portfolios.

---

## Future Improvements

Possible extensions include:

- Comparing Random Forest with Logistic Regression and Gradient Boosting models
- Probability calibration
- SHAP-based model interpretability
- Cost-sensitive learning
- More extensive cross-validation
- External validation on another credit dataset
- Feature selection
- Model deployment using Flask or FastAPI
- Building a web interface for credit-risk prediction
- Monitoring model performance after deployment

---

## Author

**Saiesh Kadam**

Electronics and Communication Engineering  
Visvesvaraya National Institute of Technology (VNIT), Nagpur

---

## Disclaimer

This project is developed for educational and experimental purposes. It should not be used as a standalone system for real-world lending or financial decision-making without appropriate validation, regulatory review, fairness assessment, and risk-management controls.
