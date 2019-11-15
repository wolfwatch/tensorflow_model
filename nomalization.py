import csv
import os
from collections import OrderedDict
from time import sleep

import pandas as pd                   # csv 불러오기
import matplotlib as mpl              # 폰트
import matplotlib.pyplot as plt       # 그래프
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 그래프 그리는 함수
def graph(word):
    # 폰트 바꿔야 한글 나옴
    plt.rc('font', family='Malgun Gothic')
    # # csv 파일 불러오기
    # df = pd.read_csv(word+'_sum.csv')
    # # 불러온 데이터 확인
    # print(df)
    # # 그래프 생성
    # df.plot(title=word, x='기간')
    # # 표시
    # plt.show()
    #
    # exit(5)

    df = pd.read_csv('./data/' + word + '_sum.csv')

    normalized_data = {}
    for coln in df.columns[1:]:

        col = df[coln]  # csv에서 읽어온 칼럼
        ncol = []  # 정규화된 칼럼
        maxv = col[len(col) - 1]  # 각 칼럼의 최대값 (TODO: 틀릴수도 있음)

        # 정규화
        if maxv == 0:
            # 칼럼의 최대값이 0일 경우 0으로 나누기를 피하기 위해 1로 변경
            maxv = 1
        for i in range(len(col)):
            # 정규화된 사용량 데이터를 ncol에 추가
            # ncol.append([i / (len(col) - 1), col[i] / maxv])
            # 앞뒤값과 평균 내서 부드럽게
            if i == 0:
                ncol.append((col[i] + col[i + 1]) / 2 / maxv)
            elif i == len(col) - 1:
                ncol.append((col[i - 1] + col[i]) / 2 / maxv)
            else:
                ncol.append((col[i - 1] + col[i] + col[i + 1]) / 3 / maxv)

        # 만들어진 ncol을 normalized_data에 저장
        normalized_data[coln] = ncol

    # df = pd.DataFrame(normalized_data)
    sc_df = pd.DataFrame(normalized_data, index=df["기간"])
    print(sc_df)
    sc_df.plot(title=word)
    plt.show()


# 읽을 단어 목록
word_list = ["따흐흑", "뚝배기", "실화냐", "팩폭","혼모노"]

for word in word_list:
    graph(word)

