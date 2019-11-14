import numpy as np
import matplotlib.pyplot as plt

##########################선언부################################
header = []
sum_list = []
only_sum_list = [] #날짜 없앤 sum_list
speed_list = []
ss_list = []
term = 1; #속도 구할 때 쓰는 날짜 간격  (오늘 누적치 - 이전 누적치) / term = 속도
w_list = ["따흐흑", "뚝배기", "실화냐", "팩폭", "혼모노"] # 검색할 단어 리스트
check = 0; #반복문에서 사이트 리스트 한번만 보여주게 하기 위해

############1.파일을 열어서 list형태로 data를 가지고 온다. header와 value를 분리한다.

def make_only_sum_list(w_name):
    global  header
    line_counter = 0
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
def make_ss_graph(s_name) :

    s_i = header.index(s_name) - 1# 사용자로부터 입력받은 사이트의 index출력 -1은 기간때문에 하나 더 더해져서 넣음

    for item in ss_list:
        plt.scatter(item[s_i][0], item[s_i][1], marker='o') #x가 누적치, y가 속도


######그래프 설정 초기화
def init_plot():
    plt.title("SPEED and SUM Graph")
    plt.grid(True)
    plt.xlabel("SUM")
    plt.ylabel("SPEED")

#####그래프 보여주기
def plt_show():
    plt.show()

#####모든 리스트 초기화
def init_list():
    global  header
    global  sum_list
    global  only_sum_list
    global  speed_list
    global  ss_list
    header = []
    sum_list = []
    only_sum_list = []  # 날짜 없앤 sum_list
    speed_list = []
    ss_list = []

###실행부###

#for 문으로 전체 반복
init_plot() # 그래프 초기 설정
s_name = ""
for w_name in w_list :
    make_only_sum_list(w_name)
    make_speed_list(only_sum_list)
    make_ss_list()
    if check == 0 :
        print("단어 리스트")
        print(w_list)
        print("사이트 리스트")
        print(header)
        s_name = input("그래프가 보고싶은 사이트를 입력하세요 : ")
        check = 1
    make_ss_graph(s_name)
    init_list()
plt_show() #그래프 그리기
