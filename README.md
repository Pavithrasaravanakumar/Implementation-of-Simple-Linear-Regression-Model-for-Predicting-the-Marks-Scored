# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Pavithra.S
RegisterNumber: 212223220073
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset = pd.read_csv("C:/Users/admin/Downloads/ML EX 2/student_scores.csv")
print(dataset.head())
print(dataset.tail())
```
![image](https://github.com/user-attachments/assets/0db8208a-f12b-420a-8223-ad41270d6444)
```
dataset.info()
```
![image](https://github.com/user-attachments/assets/9c0a01f5-bd90-4ba8-8c3b-dd95214c708f)
```
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,-1].values
print(y)
```
![image](https://github.com/user-attachments/assets/bd8654e8-c20f-4ea5-843b-1bb5e09af757)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
```
![image](https://github.com/user-attachments/assets/369ac638-0a66-4850-a1e7-60bedde39e24)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
x_test.shape
```
![image](https://github.com/user-attachments/assets/4f1e4ced-4b53-42da-bd60-bd605a3c143a)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
x_test.shape
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
```
![image](https://github.com/user-attachments/assets/ff7aa698-2d34-4da0-9bfa-0e7d265ea50f)
```
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)
```
![image](https://github.com/user-attachments/assets/ba3bc206-898c-4912-af9c-664a4d5bd837)
```
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,reg.predict(x_train),color="red")
plt.title('Traning set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,reg.predict(x_test),color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
```
```
mse = mean_squared_error(y_test, y_pred)
print('MSE = ', mse)
```
![image](https://github.com/user-attachments/assets/91f51a8a-ed75-40f2-96cd-7a20938abec7)
```
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
```
![image](https://github.com/user-attachments/assets/0af575ec-5b54-4af7-bd18-7a6ed503c51a)
```
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
![image](https://github.com/user-attachments/assets/c0993847-aa1d-4813-b6db-e68f421d6d43)


## Output:
## Training Set:
![image](https://github.com/user-attachments/assets/0c64661c-9a0c-4c66-bfd7-997d7edfdb52)

## Testing Set:
![image](https://github.com/user-attachments/assets/1cc2e814-43f9-497e-95ef-773e5f027d24)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
