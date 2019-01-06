import pandas as pd 
data = pd.read_csv("USA_Housing.csv")  

x=data.iloc[:,1:6].values
y=data.Price.values

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=0)

reg = ElasticNet()

reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)


from sklearn import metrics
import numpy as np


print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))




