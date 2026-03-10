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
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", 
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
x.head() #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt 
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['stayed','left'],filled=True)
plt.show()
```

## Output:
<img width="1379" height="751" alt="ml03" src="https://github.com/user-attachments/assets/714ba5d7-f2ee-475e-a97c-d91c2eb39beb" />
<img width="1384" height="782" alt="ml04" src="https://github.com/user-attachments/assets/c4c7986d-caeb-442d-89a2-e9d61c303ad4" />
<img width="1381" height="826" alt="ml05" src="https://github.com/user-attachments/assets/729b1f7a-0805-4a42-ae41-bc670bed825f" />
<img width="1406" height="892" alt="ml06" src="https://github.com/user-attachments/assets/524ceb05-152d-492e-948c-09a5a8416cc7" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
