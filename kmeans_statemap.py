import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import math

plt.rc('font', family='Malgun Gothic')

# csv 읽기
df = pd.read_csv('./sum/워라벨_sum.csv', parse_dates=[0], dtype=np.float64)

# 정규화된 데이터를 저장할 딕셔너리
# { 'col1': [[0, 0], [0.2, 0.2], ...], 'col2': [[], [], ...], ... }
normalized_data = {}
# 각 칼럼(사이트)별로 실행할 텐서를 저장할 딕셔너리
# { 'col1': [update_centroids, centroids, points, assignments], 'col2': [], ... }
col_tensors = {}

# 초기 centroid 값들
raw_centroids = [[0, 0], [0.25, 0], [0.5, 0.125], [0.75, 0.5], [1, 1]]
# 시험용
#raw_centroids = [[0, 0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5], [0.6, 0.6], [0.7, 0.7], [0.8, 0.8], [0.9, 0.9]]

# 기간을 제외한 각 칼럼별로
for coln in df.columns[1:]:
    col = df[coln]              # csv에서 읽어온 칼럼
    ncol = []                   # 정규화된 칼럼
    maxv = col[len(col) - 1]    # 각 칼럼의 최대값 (TODO: 틀릴수도 있음)

    # 정규화
    if maxv == 0:
        # 칼럼의 최대값이 0일 경우 0으로 나누기를 피하기 위해 1로 변경
        maxv = 1
    for i in range(len(col)):
        # 정규화된 사용량 데이터를 ncol에 추가
        #ncol.append([i / (len(col) - 1), col[i] / maxv])
        # 앞뒤값과 평균 내서 부드럽게
        if i == 0:
            ncol.append([i / (len(col) - 1), (col[i] + col[i + 1]) / 2 / maxv])
        elif i == len(col) - 1:
            ncol.append([i / (len(col) - 1), (col[i - 1] + col[i]) / 2 / maxv])
        else:
            ncol.append([i / (len(col) - 1), (col[i - 1] + col[i] + col[i + 1]) / 3 / maxv])
    
    # 만들어진 ncol을 normalized_data에 저장
    normalized_data[coln] = ncol

    # 텐서 생성
    points = tf.constant(ncol, dtype=np.float64)
    centroids = tf.Variable(raw_centroids, dtype=np.float64)

    points_expanded = tf.expand_dims(points, 0)
    centroids_expanded = tf.expand_dims(centroids, 1)

    distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
    assignments = tf.argmin(distances, 0)
    
    means = []
    for c in range(len(raw_centroids)):
        means.append(tf.reduce_mean(
            tf.gather(points, 
                    tf.reshape(
                        tf.where(
                        tf.equal(assignments, c)
                        ),[1,-1])
                    ),reduction_indices=[1]))

    new_centroids = tf.concat(means, 0)
    update_centroids = tf.assign(centroids, new_centroids)

    # 만들어진 칼럼 텐서들을 col_tensors에 저장
    col_tensors[coln] = [update_centroids, centroids, points, assignments]

# 세션 실행 후 최종 centroid 값들을 저장할 딕셔너리
# { 'col1': [[0, 0], [0, 0], ...], 'col2': [[0, 0], [0, 0], ...], ... }
final_centroids = {}

with tf.Session() as sess:
    # 세션 초기화
    sess.run(tf.global_variables_initializer())
    for coln, col_ten in col_tensors.items():   # 클러스터링을 칼럼별로 수행
        for step in range(100): # 반복 횟수
            # 세션 수행
            [_, col_centroid_values, col_points_values, col_assignment_values] = sess.run(col_ten)
        
        # 현재 칼럼에서 클러스터링이 올바르게 수행되었는지 확인
        state = {}
        for i in range(len(raw_centroids)):
            state[i] = []
        for i in range(len(col_points_values)):
            state[col_assignment_values[i]].append(col_points_values[i][0])
        
        state_invalid = False
        for k, v in state.items():
            if len(v) == 0:
                state_invalid = True
                break
        
        # 클러스터링이 올바르게 수행되지 않았을 경우 현재 칼럼을 저장하지 않음
        if state_invalid == False:
            # 현재 칼럼의 최종 centroid 값을 저장할 배열 생성
            final_centroids[coln] = []

            # 최종 centroid 값 저장
            for i in col_centroid_values:
                final_centroids[coln].append(i)
    
            # 데이터 plot하기
            plt.scatter(col_points_values[:, 0], col_points_values[:, 1], c=col_assignment_values, s=50, alpha=0.5)
            plt.plot(col_centroid_values[:, 0], col_centroid_values[:, 1], 'kx', markersize=10)

            # BEGIN 그래프 칼럼별로 보기 (주석 해제하여 사용)
            #for i in raw_centroids:
            #    plt.plot(i[0], i[1], 'rx', markersize=10)
            #plt.title(coln)
            #plt.show()
            #plt.clf()
            # END 그래프 칼럼별로 보기

# BEGIN 전체 그래프 보기 (칼럼별로 보기 부분 주석 처리 필수)
for i in raw_centroids:
    plt.plot(i[0], i[1], 'rx', markersize=10)
plt.show()
# END 전체 그래프 보기

def generate_spread_matrix(centroids):
    # centroids: { 'col1': [[0, 0], [0, 0], ...], 'col2': [[0, 0], [0, 0], ...], ... }
    
    # 이 함수에서 사용할 centroid 값 딕셔너리
    n_centroids = {}
    # 처음과 마지막 centroid 제거 (부정확)
    for k, v in centroids.items():
        n_centroids[k] = v[1:-1]
    
    spread_matrix = {}

    for site_current, centroid_current in n_centroids.items():
        if site_current not in spread_matrix:
            spread_matrix[site_current] = {}
        matrix_current = spread_matrix[site_current]
        for site_against, centroid_against in n_centroids.items():
            if site_current == site_against:
                matrix_current[site_against] = 0
                continue
            # 사용 가능한 변수
            deltax = centroid_against[0] - centroid_current[0]
            currenty = centroid_current[1]
            #matrix_current[site_against] = ... FIXME

    return spread_matrix

# 확산 행렬 생성
spread_matrix = generate_spread_matrix(final_centroids)