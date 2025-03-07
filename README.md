# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Prepare the Data
2. Calculate the Slope (m) and Intercept (b)
3. Make Predictions 
4. Evaluate and Display Results

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Kamalesh y
RegisterNumber: 212223243001

```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
```
## OUTPUT:
![Screenshot 2025-03-02 151632](https://github.com/user-attachments/assets/c72e96e9-f0d3-440c-9935-a3f57777ee6e)

```
df.tail()
```
## OUTPUT:
![Screenshot 2025-03-02 151646](https://github.com/user-attachments/assets/e97650b9-ea5b-4d74-8d6d-8f439c7220d0)
```
x=df.iloc[:,:-1].values
x
```
## OUTPUT:
![Screenshot 2025-03-02 151701](https://github.com/user-attachments/assets/a6677696-b999-4a90-b2f0-f099df04f41f)
```
y=df.iloc[:,1].values
y
```
## OUTPUT:
![Screenshot 2025-03-02 151717](https://github.com/user-attachments/assets/16107a67-337d-4943-ab4d-516980569575)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
```
```
y_test
```
## OUTPUT:
![Screenshot 2025-03-02 151748](https://github.com/user-attachments/assets/2fa53534-eeeb-4e1c-a6b6-bdb0db312f09)
```
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores (Trainind Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## OUTPUT:
![Screenshot 2025-03-02 151813](https://github.com/user-attachments/assets/5fb816e2-04b6-490d-9154-4f1b676f3684)
```
plt.scatter(x_train,y_train,color="purple")
plt.plot(x_train,regressor.predict(x_train),color='yellow')
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## OUTPUT:
![Screenshot 2025-03-02 151827](https://github.com/user-attachments/assets/5a14ccec-884e-40e0-8129-ccfe3e2b2437)
```
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mse)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```
## OUTPUT:
![Screenshot 2025-03-02 151840](https://github.com/user-attachments/assets/6774a99e-fbc8-442e-baf8-6914a3129f91)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
