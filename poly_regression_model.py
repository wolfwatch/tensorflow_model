import operator

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_csv('./data/centroid_after.csv')
xy = np.array(df.iloc[:, :], dtype=np.float32)
# x : 시간
x = xy[:, :-1]
# y : 누적량
y = xy[:, [-1]]


polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

# 모델 저장
joblib.dump(model, 'centroid.pkl')

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y, y_poly_pred)
print(rmse)
print(r2)

plt.scatter(x, y, s=10)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='m')
plt.show()

# 저장된 모델 불러와서 테스트
k = 5
step = float(1 / (k - 1))
x = []
x_temp = 0
for i in range(k):
    temp = []
    temp.append(step * i)
    x.append(temp)
    x_temp += step

polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(x)
clf_from_joblib = joblib.load('centroid.pkl')
y = clf_from_joblib.predict(x_poly)
print(y)

raw_centroids = []
for i in range(k):
    y_temp = y[i][0]/y[k-1][0]
    raw_centroids.append([x[i][0], y_temp])

print(raw_centroids)

