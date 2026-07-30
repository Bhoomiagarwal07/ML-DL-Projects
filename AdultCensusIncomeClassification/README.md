# Adult Census Income Classification

## 📌 Objective
Predict whether an individual's annual income exceeds **$50K** based on 1994 U.S. Census
demographic and employment data, using both Logistic Regression and Random Forest
classifiers.

## 📊 Dataset
**Adult Census Income Dataset** (48,842 rows, 14 columns)
Source: [UCI Machine Learning Repository — Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
(also mirrored on [Kaggle](https://www.kaggle.com/datasets/uciml/adult-census-income))

*(The dataset is not uploaded to this repo — the notebook loads it automatically from a public mirror.)*

## 🛠️ Libraries Used
- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `scikit-learn` — train/test split, feature scaling, Logistic Regression, Random Forest, evaluation metrics
- `matplotlib` / `seaborn` — visualization (confusion matrices, feature importance plot)

## 🔍 Methodology
1. **Data Understanding** — loaded the dataset, identified numerical and categorical
   features and the target (`class`: `<=50K` or `>50K`), and checked class balance (~76%/24%
   — moderately imbalanced).
2. **Data Preprocessing** — discovered and handled the dataset's classic quirk where missing
   values are encoded as `"?"` rather than true NaN, imputed missing categorical values with
   the mode, encoded the target and one-hot encoded categorical features, split 80/20 with
   stratification, and standardized numerical features.
3. **Model Development** — trained two models on the same data: `LogisticRegression` and a
   `RandomForestClassifier` with 200 estimators.
4. **Model Evaluation & Comparison** — evaluated both using Accuracy, Precision, Recall, and
   F1-Score, visualized confusion matrices for both, and generated a Random Forest feature
   importance plot.

## 📈 Results

| Metric | Logistic Regression | Random Forest |
|--------|----------------------|-----------------|
| Accuracy  | ≈ 85.2% | ≈ 85.1% |
| Precision | ≈ 74.0% | ≈ 72.0% |
| Recall    | ≈ 59.0% | ≈ 62.1% |
| F1-Score  | ≈ 65.7% | ≈ 66.7% |

**Key finding:** `age`, `hours-per-week`, `capital-gain`, marital status, and education level
are the strongest predictors of income exceeding $50K.

## ✅ Conclusion
This project built and compared Logistic Regression and Random Forest classifiers to predict
whether an individual's income exceeds $50K using 1994 U.S. Census demographic and employment
data. After handling the dataset's non-standard `?` missing-value marker (imputing missing
categorical values with the mode), one-hot encoding categorical features, and standardizing
numerical features, both models achieved comparable accuracy around 85%, with Random Forest
showing a modest edge in recall and F1-score for the higher-income class. Age, weekly working
hours, capital gains, marital status, and education level emerged as the strongest predictors
of income, aligning with real-world intuition about career stage and earning potential. A key
challenge in this dataset is its class imbalance (~76% earning `<=50K` vs. ~24% earning
`>50K`), which caused both models to somewhat under-predict the minority high-income class —
a limitation that techniques like class weighting, oversampling (e.g. SMOTE), or threshold
tuning could help address in a more refined version of this project.

## 📂 Files
- `AdultCensusIncomeClassification.ipynb` — full notebook with code, outputs, and visualizations
