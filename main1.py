import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgbm
import xgboost as xg
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import csv

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Read the datasets
data = pd.read_csv("train.csv")
data1 = pd.read_csv("test.csv")
gender_submission = pd.read_csv("gender_submission.csv")

label_encoder = LabelEncoder()
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if str(big_string).find(substring) != -1:
            return substring
    return np.nan

title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']

data['Title'] = data['Name'].map(lambda x: substrings_in_string(x, title_list))
data1['Title'] = data1['Name'].map(lambda x: substrings_in_string(x, title_list))

def replace_titles(x):
    title = x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Dr':
        if x['Sex'] == 'Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title

data['Title'] = data.apply(replace_titles, axis=1)
data1['Title'] = data1.apply(replace_titles, axis=1)

# Turning cabin number into Deck
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
data['Deck'] = data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
data['Family_Size'] = data['SibSp'] + data['Parch']
data['Age*Class'] = data['Age'] * data['Pclass']
data['Fare_Per_Person'] = data['Fare'] / (data['Family_Size'] + 1)


data1['Deck'] = data1['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
data1['Family_Size'] = data1['SibSp'] + data1['Parch']
data1['Age*Class'] = data1['Age'] * data1['Pclass']
data1['Fare_Per_Person'] = data1['Fare'] / (data1['Family_Size'] + 1)
print(data.dtypes)

objects_list = ["Name", "Sex", "Ticket", "Cabin", "Embarked", "Title", "Deck"]
for i in objects_list:
    data[i] = label_encoder.fit_transform(data[i])
    data1[i] = label_encoder.fit_transform(data1[i])

data = data.query("Age.notnull()")

ss = StandardScaler()
x, y = data.drop(columns='Survived', axis=1), data['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42,shuffle=True)

#x_test = ss.transform(x_test.values)
def compare_models(x):
    x.fit(x_train, y_train)
    yhat = x.predict(x_test)
    y_known = x.predict(x_train)
    algoname = x.__class__.__name__
    accuracy = round(accuracy_score(y_test, yhat), 3)
    accuracy_train = round(accuracy_score(y_train, y_known), 3)
    precision = round(precision_score(y_test, yhat), 2)
    recall = round(recall_score(y_test, yhat), 2)
    f1 = round(f1_score(y_test, yhat), 2)
    return algoname, accuracy, accuracy_train, precision, recall, f1

algo = [
    GradientBoostingClassifier(),
    lgbm.LGBMClassifier(),
    xg.XGBRFClassifier(),
    xg.XGBClassifier(),
    SGDClassifier(),
    LogisticRegression()
]

score = []
for a in algo:
    score.append(compare_models(a))

# Collate all scores in a table
column_names = ['Model', 'Accuracy', 'Accuracy on Train', 'Precision', 'Recall', 'F1 Score']
print(pd.DataFrame(score, columns=column_names))

# LGBMClassifier
LGBMClassifier = lgbm.LGBMClassifier()
LGBMClassifier.fit(x_train, y_train)

y_pred_test = LGBMClassifier.predict(data1)

# Save to a CSV file without using pandas
with open("predictions_result.csv", "w", newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write header
    csv_writer.writerow(['PassengerId', 'Survived'])

    # Write data
    for passenger_id, prediction in zip(gender_submission['PassengerId'], y_pred_test):
        csv_writer.writerow([passenger_id, prediction])

print("Result predictions saved to 'predictions_result.csv'")
