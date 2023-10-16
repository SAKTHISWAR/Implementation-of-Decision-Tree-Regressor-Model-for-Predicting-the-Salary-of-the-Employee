# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee
### AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1.Hardware – PCs
2.Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1.Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.calculate Mean square error,data prediction and r2.


# Program:
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SAKTHISWAR S
RegisterNumber:  212222230127


```python

import pandas as pd
data = pd.read_csv("/content/Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Position'] = le.fit_transform(data['Position'])
data.head()

x = data[['Position','Level']]
y = data['Salary']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_pred)
mse

r2 = metrics.r2_score(y_test, y_pred)
r2

dt.predict([[5,6]])

```
## Output:
### data.head()
![image](https://github.com/SAKTHISWAR/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/blob/main/m1.png)

### data.info()
![image](https://github.com/SAKTHISWAR/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/blob/main/m2.png)


### isnull().sum()
![image](https://github.com/SAKTHISWAR/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/blob/main/m3.png)

### data.head() for salary
![image](https://github.com/SAKTHISWAR/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/blob/main/m4.png)


### MSE value
![image](https://github.com/SAKTHISWAR/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/blob/main/m5.png)


### R2 value
![image](https://github.com/SAKTHISWAR/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/blob/main/m6.png)


### Prediction value
![image](https://github.com/SAKTHISWAR/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/blob/main/m7.png)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
