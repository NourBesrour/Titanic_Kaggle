
# Titanic Survival Prediction

This project is a machine learning classification task aimed at predicting the survival of passengers on the Titanic using various machine learning models. The dataset is based on passenger information such as age, gender, fare, and title, among other features. Multiple classification algorithms have been used to predict survival, and their performance has been compared based on accuracy, precision, recall, and F1 score.

## Project Overview

The Titanic dataset is one of the most popular datasets for binary classification tasks. The goal of this project is to predict whether a passenger survived or not (`Survived` column: 1 = Survived, 0 = Did not survive) based on features like passenger class, name, gender, age, fare, and other attributes.

## Dataset

- **train.csv**: Training dataset containing features and the target `Survived` column.
- **test.csv**: Test dataset without the `Survived` column.
- **gender_submission.csv**: Sample file showing how the prediction results should be structured.

### Feature Engineering

Several new features have been created to enhance model performance:

1. **Title Extraction**: Titles such as 'Mr', 'Mrs', 'Miss', etc., have been extracted from passenger names.
2. **Deck**: Derived from the cabin number.
3. **Family Size**: Combination of `SibSp` (siblings/spouses aboard) and `Parch` (parents/children aboard).
4. **Age*Class**: Product of age and class to account for socio-economic factors.
5. **Fare Per Person**: Fare divided by family size to capture relative fare contribution per person.

### Preprocessing

- Categorical columns such as `Sex`, `Embarked`, `Title`, and `Deck` have been label-encoded.
- Missing values for the `Age` feature have been handled by removing rows with null values for this project.
- Features have been scaled using `StandardScaler`.

## Models Used

The following models were compared in terms of performance:

1. **Gradient Boosting Classifier**
2. **LightGBM Classifier**
3. **XGBoost RF Classifier**
4. **XGBoost Classifier**
5. **SGD Classifier**
6. **Logistic Regression**

## Model Evaluation

Each model was evaluated using the following metrics:

- **Accuracy**: Proportion of correctly classified instances.
- **Precision**: Proportion of true positives out of all positive predictions.
- **Recall**: Proportion of true positives out of all actual positives.
- **F1 Score**: Harmonic mean of precision and recall.

The models were compared using an 80/20 train-test split with random shuffling.

## Predictions

Once the models were trained and evaluated, the **LightGBM Classifier** was used to make predictions on the test dataset (`test.csv`). The predictions were saved to a CSV file named `predictions_result.csv` in the required format.

## How to Run

1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the script:
   ```bash
   python titanic_ml_classification.py
   ```

   This will train the models, evaluate them, and save the predictions in the `predictions_result.csv` file.

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- lightgbm
- xgboost
- CSV library

To install the dependencies, use the following command:
```bash
pip install pandas numpy scikit-learn lightgbm xgboost
```

## File Descriptions

- `train.csv`: Training data containing passenger features and survival status.
- `test.csv`: Test data without the survival status (to be predicted).
- `gender_submission.csv`: Sample file for submission format.
- `main1.py`: Main script to preprocess data, train models, and save predictions.
- `predictions_result.csv`: Output file containing predictions for the test dataset.
  
## Acknowledgments

This project is based on the Titanic dataset from Kaggle, a popular competition to practice classification tasks.
