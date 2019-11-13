from collections import OrderedDict

import pandas as pd
import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv

def make_chart_data(word):
    X = tf.placeholder(tf.float32, shape=[None, 2])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    W = tf.Variable(tf.random_normal([2, 1]), name="weight")
    b = tf.Variable(tf.random_normal([1]), name="bias")

    hypothesis = tf.matmul(X, W) + b

    # 데이터를 파라미터로 받음
    # input_data = [110, 100]

    model = tf.global_variables_initializer()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(model)

        # 1. 데이터를 읽음 : word_sum.csv 읽음
        data_df = pd.read_csv('./sum/'+word+'_sum.csv')
        result_dict = OrderedDict()

        result_dict["date"] = []
        date_list = data_df["기간"]
        for date in date_list[1:]:
            result_dict["date"].append(date)
        print(result_dict)

        site_list = data_df.columns[1:]
        # 2. 컬럼마다(사이트마다) 값을 읽음 -> a_data = { counts[i] = [], speed[i] = [] }
        for site in site_list:
            temp = []
            save_path = "./models/" + site + ".cpkt"
            saver.restore(sess, save_path)
            for i in range(1, len(data_df)):
                # 3. 이 정보를 a_data = [counts[i], speed[i]]로 두고서
                counts = data_df.loc[i, site]
                speed = (data_df.loc[i, site] - data_df.loc[i - 1, site]) / 30
                a_data = [counts, speed]
                # 4. tf_model()에 a_data에 대한 각각의 결과를 -> state[i]에 append함

                print("----------model_test-----------")
                print(str(a_data[0]) + " // " + site)

                # 2차원 배열이기때문에 이런식으로 표현 가능
                data = ((a_data[0], a_data[1]),)
                arr = np.array(data, dtype=np.float32)
                x_data = arr[0:1]  # input 의 크기
                predict = sess.run(hypothesis, feed_dict={X: x_data})
                print("predict : " + str(predict[0]))

                temp.append( predict[0] )

            # temp["state"]는 result_df의 한 컬럼([site])을 차지해야한다.
            print(len(temp))
            print(str(len(temp))+" : "+str(len(result_dict["date"])))
            result_dict[site] = temp
            # 4. result_df에 모아서, 사이트_data.csv에다가 저장
            result_df = pd.DataFrame(result_dict)
            print("result")
            print(result_df) # 컬럼이 점점 증가하는, 데이터가 횡으로 길어지는 모습이 보여야함.
        # 5. 사이트_word_result = ["date", "counts", "speed", "state"]
        # 6. word_result = [ 컬럼 : 사이트별 "state", 인덱스 : "date"]
        #   { date, 사이트1, 사이트2, 사이트3, 사이트4,..................}
        #   { 2019/08, 12, 11, 2, 100, 0 ,0 .........}

        # 즉, 한 컬럼을 완성하기 위해서, 2~4 과정을 row의 갯수만큼 반복함
        # 7. 여러컬럼, 여러 사이트들에 대해서 반복함

        # 결과를 그래프로 그릴수 있는, csv로 저장.
        result_df.to_csv('./chart/'+word + "_chart.csv")


make_chart_data("팩폭")
