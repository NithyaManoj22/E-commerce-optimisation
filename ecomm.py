import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import seaborn as sns

path = "C:/Users/User/Desktop/ecomm.csv"
ecom = pd.read_csv(path)
print(ecom.head(10))

correlations = ecom.corr()['Yearly Amount Spent'].drop('Yearly Amount Spent')

print(correlations)

print(ecom.describe())

print(ecom.groupby('Yearly Amount Spent').size())

sns.pairplot(ecom)
plt.show()
sns.lmplot('Yearly Amount Spent', 'Length of Membership', data=ecom)
plt.show()
array=ecom.values
x=array[:,3:7]
y=array[:,7]

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3,random_state =3)

reg=LinearRegression()
reg.fit(x_train, y_train)
print(reg.coef_)
train_pred = reg.predict(x_train)
test_pred=reg.predict(x_test)
print(test_pred)
print("Accuracy due to linear regression test is :(:.2f)",format(reg.score(x_train,y_train)))


train_rmse = mean_squared_error(train_pred, y_train)**0.5
print(train_rmse)
test_rmse=mean_squared_error(test_pred, y_test)**0.5
print (test_rmse)
# rounding off the predicted values for test set
predicted_data = np.round_(test_pred)
print (predicted_data)
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,test_pred))
print('Mean_Squared_Error:', metrics.mean_squared_error(y_test,test_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,test_pred)))

coefficients=pd.DataFrame(reg.coef_,correlations.abs())
coefficients.columns = ['Coefficient']
print(coefficients)
