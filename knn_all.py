from time import sleep

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cp_list = []
state_list = []
####data_set함수를 위한 data_list와 category를 만드는 함수
def make_list(site):
    global header
    line_counter = 0
    # site = input("KNN을 적용할 사이트를 입력하세요(_data.csv 생략): ")
    with open('./data/'+site + "_data.csv", 'rt', encoding='UTF8') as f:
        while 1:
            data = f.readline().replace("\n", "")
            # data에 파일을 한 줄씩 불러옴
            if not data:
                break
            if line_counter == 0:
                header = data.split(",")  # 맨 첫 줄은 header로 저장
            else:
                temp_stl = data.split(",")
                #print(temp_stl)
                cp_list.append([float(temp_stl[1]),float(temp_stl[2])]) #int로 변환하여 cp_list state_list저장
                if float(temp_stl[1]) == 0 : # count가 0인 경우
                    state_list.append(float(0)) # 정체
                else : #state값이 0.89...이런식으로 나오는게 몇개 있음
                    state_list.append(round(float(temp_stl[3]) * 10 / float(temp_stl[1])))
                    #정수화 하여 삽입

            line_counter += 1

###test용 데이터를 입력받아
def data_set(data_list, category):
    dataset = np.array(data_list)  # 분류집단
    size = len(dataset)
    # class_target = np.tile(target, (size, 1))  # 분류대상
    class_category = np.array(category)  # 분류범주

    return dataset, class_category # dataset, class_target, class_category

def classify(dataset, class_target, class_category, k):
    # 유클리드 거리 계산
    diffMat = class_target - dataset  # 두 점의 차
    sqDiffMat = diffMat ** 2  # 차에 대한 제곱
    row_sum = sqDiffMat.sum(axis=1)  # 차에 대한 제곱에 대한 합
    distance = np.sqrt(row_sum)  # 차에 대한 제곱에 대한 합의 제곱근(최종거리)

    # 가까운 거리 오름차순 정렬
    sortDist = distance.argsort()

    # 이웃한 k개 선정
    class_result = {}
    for i in range(k):
        c = class_category[sortDist[i]]
        class_result[c] = class_result.get(c, 0) + 1

    return class_result



# 분류결과 출력 함수 정의
def classify_result(class_result):
    tree = two = one = none = 0
    dic = {3: "일시적 유행", 2: "확산", 1: "일반", 0: "정체"}
    for c in class_result.keys():
        if c == 3:
            tree = class_result[c]
        elif c == 2:
            two = class_result[c]
        elif c == 1:
            one = class_result[c]
        else :
            none = class_result[c]

    a = np.array([none, one, two, tree])
    return dic[np.argmax(a)]

# 그래프 출력
def draw_knngraph(data_list, state_list):
    plt.clf() #초기화
    plt.title("SPEED and SUM Graph")
    plt.grid(True)
    plt.xlabel("SUM")
    plt.ylabel("SPEED")
    for i in range(len(data_list)) :
        if state_list[i] == 0 :
            plt.scatter(data_list[i][0], data_list[i][1], marker='o',color ='black') #정체
        elif state_list[i] == 1 :
            plt.scatter(data_list[i][0], data_list[i][1], marker='D', color='blue') #일반
        elif state_list[i] == 2:
            plt.scatter(data_list[i][0], data_list[i][1], marker='p', color='magenta')#확산
        else :
            plt.scatter(data_list[i][0], data_list[i][1], marker='*', color='red')#일시적 유행

    #plt.scatter(target[0], target[1], marker='X', color='orange')
    plt.show()


def k_test(target_list):
    for k_num in range(1, 11):
        match=0 #일치 갯수
        for index, row in target_list.iterrows():
            target = [float(row['counts']), float(row['speed'])]
            k = k_num
            class_result = classify(dataset, np.array(target), class_category, k)
            dic = {3: "일시적 유행", 2: "확산", 1: "일반", 0: "정체"}
            if dic[row['state']] == classify_result(class_result): #예상 상태와 같은 값이 나오면 match+
                match += 1
        print("k값이", k_num, "일 때::", "정확도는::", (match/len(target_list))*100,"%") #10은 k_num의 순환수


#############
header = []
####실행부#####

# # 분류대상
# c = float(input('데이터의 count를 입력하세요 :'))
# s = float(input('데이터의 speed를 입력하세요 :'))
# target = [c, s]

site_list = ["네이트판", "도탁스", "아프리카TV", "오유", "웃긴대학", "이종락싸", "인벤", "인스티즈", "일베", "쭉빵여시", "트위터"]

for site in site_list:
    #numpy화하여 데이터 셋 처리할 list 생성
    make_list(site)
    # dataset 생성
    dataset, class_category = data_set(cp_list, state_list)
    #print(state_list)

    #예상 상태 알아보는 부분
    # k = int(input('k값 입력(1~10):'))
    # class_result = classify(dataset, class_target, class_category, k)  # classify()함수호출
    # print(class_result)
    # a = classify_result(class_result)
    # print(a + "상태입니다.")
    draw_knngraph(cp_list, state_list)

    #정확도 테스트 부분
    # target_list = [[170, 1, 1],[118,0.566667,1],[81,0.433333,1],[82,1.333333,2],[942,7.1,3]]
    # print("")
    # print(target_list)
    df = pd.read_csv('./test/' + site + '_test.csv')
    target_list = df.iloc[:, :]
    print(target_list)
    print("**위의 리스트에 대한 사이트 " + site +"의 정확도 출력**")
    k_test(target_list)
    cp_list = []
    state_list = []


