# Credit Scoring Model Using Machine Learning


## Project Overview

This project develops and compares machine learning classification models to predict credit risk using historical financial and applicant information.

The project uses the German Credit dataset from the UCI Machine Learning Repository.

## Objective

To classify applicants into Good Credit Risk and Bad Credit Risk categories using supervised machine learning techniques.

## Dataset Information

**Dataset:** Statlog (German Credit Data)

**Source:** https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

| Property | Details |
|---|---|
| Total Records | 1,000 |
| Original Features | 20 |
| Good Credit Risk | 700 (70%) |
| Bad Credit Risk | 300 (30%) |
| Training Records | 800 |
| Testing Records | 200 |

The dataset includes credit amount, credit duration, credit history, employment information, savings status, age, and other financial or demographic attributes.

## Data Preprocessing

The following preprocessing techniques were applied:

- Missing value and duplicate record verification
- Exploratory Data Analysis (EDA)
- Feature engineering using credit amount per month
- One-Hot Encoding of categorical variables
- StandardScaler for numerical features
- Stratified train-test split (80:20)

The preprocessing process expanded the original 21 input features (including the engineered feature) into 62 numerical features.

## Machine Learning Algorithms

Three classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **78.50%** | 67.35% | **55.00%** | **60.55%** | **0.8057** |
| Decision Tree | 72.50% | 57.14% | 33.33% | 42.11% | 0.6859 |
| Random Forest | 76.50% | 69.70% | 38.33% | 49.46% | 0.7927 |

Precision, Recall, and F1-Score refer to the Bad Credit Risk class.

## Best Performing Model

**Logistic Regression** achieved the best overall results among the three evaluated models.

- Accuracy: 78.50%
- ROC-AUC: 0.8057
- Bad Risk Recall: 55.00%

The trained model and preprocessing pipeline were saved together using Joblib.

## Prediction Example

The saved model was tested on a held-out applicant record.

**Model Prediction:** Good Credit Risk

**Estimated Bad Credit Risk Probability:** 22.76%

**Actual Class:** Good Credit Risk

The model correctly classified this sample.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- UCI ML Repository
- Google Colab

## Installation

Clone this repository:

```bash
git clone https://github.com/Suchi0012/CodeAlpha_CreditScoringModel.git
```

Navigate to the project directory:

```bash
cd CodeAlpha_CreditScoringModel
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

1. Download or clone the repository.
2. Open `CodeAlpha_CreditScoringModel.ipynb` in Google Colab or Jupyter Notebook.
3. Install the required Python libraries.
4. Execute the notebook cells sequentially.
5. Review the visualizations, model performance comparison, and predictions.

The dataset is retrieved using the `ucimlrepo` package.

## Project Structure

```text
CodeAlpha_CreditScoringModel/
├── CodeAlpha_CreditScoringModel.ipynb
├── credit_scoring_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Limitations

This project is intended for educational purposes only. The models have not undergone comprehensive fairness testing, probability calibration, or external validation.

The models should not be used to make real-world lending or credit approval decisions.

## Conclusion

This project demonstrates the application of machine learning classification algorithms to credit risk prediction.

Logistic Regression achieved the strongest overall performance among the evaluated models, with 78.50% accuracy and an ROC-AUC of 0.8057.

## Author

**Sadia Afrin Suchi**

