# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1 .Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: monesh s
RegisterNumber: 25006689
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

data = pd.read_csv("Employee (1).csv")

print("OUTPUT:\nDATA HEAD:")
print(data.head(), "\n")

print("DATASET INFO:")
print(data.info(), "\n")

print("NULL DATASET:")
print(data.isnull().sum(), "\n")

print("VALUES COUNT IN THE LEFT COLUMN:")
print(data['left'].value_counts(), "\n")

le = LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])

print("DATASET TRANSFORMED HEAD:")
print(data.head(), "\n")

X = data[['satisfaction_level', 'last_evaluation', 'number_project',
          'average_montly_hours', 'time_spend_company',
          'Work_accident', 'promotion_last_5years', 'salary']]
y = data['left']

print("X.HEAD:")
print(X.head(), "\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("ACCURACY:", accuracy, "\n")

sample = [[0.5, 0.8, 9, 260, 6, 0, 1, 2]]
prediction = dt.predict(sample)
print("DATA PREDICTION:", prediction, "\n")
```

## Output:
<img width="1019" height="619" alt="Screenshot 2026-03-10 082817" src="https://github.com/user-attachments/assets/35d2ed27-873c-4726-abe2-c371805c3792" />
<img width="504" height="681" alt="Screenshot 2026-03-10 082836" src="https://github.com/user-attachments/assets/52db4cdd-e73e-41a3-8c65-75d0b4feadee" />
<img width="1000" height="773" alt="Screenshot 2026-03-10 082856" src="https://github.com/user-attachments/assets/fe259b1a-844b-4700-a938-2af39821b637" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
