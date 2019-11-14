from time import sleep

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df = pd.read_csv('./data/인벤_data.csv')
xy = np.array(df.iloc[:, 1:], dtype=np.float32)
x_data = xy[:, :-1]
print("---스케일전 xy---")
print(x_data)

# 정규화(스케일링) 작업,
fitted = scaler.fit(x_data)
s = scaler.transform(x_data)
print(df.columns[1:3])
sc_df = pd.DataFrame(s, columns=df.columns[1:3], index=list(df.index.values))
xy = np.array(sc_df, dtype=np.float32)
x_data = xy[:, :]
print(df)
print(sc_df)

# 그려서 확인
sns.lmplot("counts", "speed", data=sc_df, fit_reg=False, size=6)
plt.show()

print("--스케일 후 xy---")
print(x_data)


# --- 학습 부분 ---
# 모든 데이터를 상수 텐서로 옮김
vectors = tf.constant(x_data)

# 클러스터의 갯수
k = 4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k,-1]))

print(vectors.get_shape())
print(centroides.get_shape())

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)

# 같은 차원 size 만듬
assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroides)), 2), 0)

# 여기가 분류의 기준이 되는 알고리즘
means = tf.concat([tf.reduce_mean(
    tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])),
                                  reduction_indices=[1])for c in range(k)], 0)

update_centroides = tf.assign(centroides, means)

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)
for step in range(1000):  # 학습? 횟수
   _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])

# assignment_values 텐서의 결과를 확인
data = {"x": [], "y": [], "cluster": []}

for i in range(len(assignment_values)):
    data["x"].append(x_data[i][0])
    data["y"].append(x_data[i][1])
    data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()


