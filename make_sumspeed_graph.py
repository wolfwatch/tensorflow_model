import numpy as np
import matplotlib.pyplot as plt
import csv

##########################선언부################################
header = []
sum_list = []
only_sum_list = [] #날짜 없앤 sum_list
speed_list = []
ss_list = []
term = 1; #속도 구할 때 쓰는 날짜 간격  (오늘 누적치 - 이전 누적치) / term = 속도


############1.파일을 열어서 list형태로 data를 가지고 온다. header와 value를 분리한다.

def make_only_sum_list():
    global  header
    line_counter = 0
    w_name = input("그래프를 그릴 신조어를 입력하세요(_sum.csv 생략): ")
    with open(w_name+"_sum.csv", 'rt', encoding='UTF8') as f:
        while 1:
            data = f.readline().replace("\n","")
            #data에 파일을 한 줄씩 불러옴
            if not data:
                break
            if line_counter == 0:
                header = data.split(",") # 맨 첫 줄은 header로 저장
            else:
                sum_list.append(data.split(","))
            line_counter += 1

     #날짜 부분을 뺍니다.
    for item in sum_list :
       only_sum_list.append(item[1:])



############2. numpy를 사용하여 속도를 구한다. 그리고 그 값을 speed_list에 넣기

def make_speed_list(only_sum_list):
    c = np.array(only_sum_list[0])

    #ssdic = {key: [] for key in header}
    for item in range(len(only_sum_list)):
        if item == 0 :
            continue
        else:
            c = (np.array(only_sum_list[item]).astype(np.int32) - np.array(only_sum_list[item-1]).astype(np.int32))/term
            speed_list.append(c.tolist())




############3. only_sum_list와 speed_list를 사용해서 ss_list얻기## ss는 sum speed 준말

def make_ss_list():
    for i in range(len(speed_list)):
        if i == 0 :
            continue
        else :
            t = np.stack((np.array(only_sum_list[i+1]).astype(np.int32), speed_list[i]), axis=-1)
            ss_list.append(t.tolist())



###########4. 그래프 그리기
def make_ss_graph() :
    s_name = input("그래프가 보고싶은 사이트를 입력하세요 : ")

    s_i = header.index(s_name) - 1# 사용자로부터 입력받은 사이트의 index출력 -1은 기간때문에 하나 더 더해져서 넣음

    for item in ss_list:
        plt.scatter(item[s_i][0], item[s_i][1], marker='o') #x가 누적치, y가 속도
    plt.title("SPEED and SUM Graph")
    plt.grid(True)
    plt.xlabel("SUM")
    plt.ylabel("SPEED")
    plt.show()



###실행부
make_only_sum_list()
make_speed_list(only_sum_list)
make_ss_list()
print("사이트 리스트")
print(header)
make_ss_graph()