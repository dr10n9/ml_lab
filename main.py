import os
import pandas as pd
#Для виконання лабораторної роботи був обраний датасет з величиною чайових
#os.chdir('drive/My Drive/Data')
data = pd.read_csv('./dataset.csv')

data["sex"] = data["sex"].astype('category')
data["smoker"] = data["smoker"].astype('category')
data["time"] = data["time"].astype('category')

data["sex"] = data["sex"].cat.codes
data["smoker"] = data["smoker"].cat.codes
data["time"] = data["time"].cat.codes
data

dfDummies = pd.get_dummies(data['day'], prefix = 'day')

data = data.drop(columns=['day'])
data = pd.concat([data, dfDummies], axis=1)
data

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.1)
Y_train = train['tip']
X_train = train.drop(columns=['tip'])
Y_test = test['tip']
X_test = test.drop(columns=['tip'])

from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
reg = linear_model.ElasticNet(random_state=0)
reg.fit(X_train, Y_train)
mean_absolute_error(reg.predict(X_test), Y_test)

print(reg.coef_)