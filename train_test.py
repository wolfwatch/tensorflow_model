from time import sleep

import pandas as pd
import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv


def train_model(site_name):
    model = tf.global_variables_initializer()

    data = read_csv('./data/'+site_name+'_data.csv', sep=',')

    df = pd.read_csv('./data/'+site_name+'_data.csv')
    xy = np.array(df.iloc[:-16, 1:], dtype=np.float32)
    # print(df.iloc[:-16, 1:])  -> count, speed, state
    # print(df.iloc[:-16,1:-1]) -> count, speed

    # x : count, speed => 파라미터
    x_data = xy[:, :-1]
    # y : state => 세타
    y_data = xy[:, [-1]]


    X = tf.placeholder(tf.float32, shape=[None, 2])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    W = tf.Variable(tf.random_normal([2, 1]), name="weight")
    b = tf.Variable(tf.random_normal([1]), name="bias")

    hypothesis = tf.matmul(X, W) + b

    ##### 여기 부터 모델 생성 부분

    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000000007)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for step in range(300001):
        cost_, hypo_, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
        if step % 500 == 0:
            print("#", step, " loss : ", cost_)
            print("- predict : ", hypo_[0])


    # 최적화 요소는 고려하지 않고 모델이 저장됨
    # saver = tf.train.Saver()
    saver1 = tf.compat.v1.train.Saver()
    save_path = saver1.save(sess, "./models/"+site_name+".cpkt")
    print(site_name+" 모델 생성")


site_list = ["네이트판", "도탁스", "아프리카TV", "오유", "웃긴대학", "이종락싸", "인벤", "인스티즈", "일베", "쭉빵여시", "트위터"]

for site in site_list:
    print("Train : " + site)
    train_model(site)
    print(site + " has been trained")
