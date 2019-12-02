import csv
import os
from collections import OrderedDict
from time import sleep

import pandas as pd
from pymongo import MongoClient
from urllib.parse import quote_plus
import traceback
from datetime import datetime, timedelta


# fast2 는 크롤링할때부터 15일치간의 누적치를 저장해둔 데이터를 DB에서 읽어올 경우에 대한 모듈
def make_cursum(db, word):
    index_date = OrderedDict()
    count_row = OrderedDict()

    zero_cursor = db.result_table.find({"word": word, "count": {"$gt": 0}}, {"_id": 0}).sort([("date", 1)])

    # 사이트 리스트 추출
    sitelist_cursor = db.word_table.find({"word": word}, {"_id": 0})
    list2 = sitelist_cursor[0]["last_info"]
    site_list = []
    for item in list2:
        site_list.append(item["site"])
        count_row[item["site"]] = []

    #for site2 in count_row:
        #print(site2)

    cnt = 0
    sum_day = 15
    index_date["date"] = []
    count = 0
    # 4. 해당 날짜를 이용해, 총 읽어야할 갯수를 확인
    list_cursor = db.result_table.find({"word": word, "site": site_list[0]},
                                       {"_id": 0, "date": 1, "count": 1}).sort([("date", 1)])
    raw_total = list_cursor.count()

    o = OrderedDict()
    o = list(list_cursor)
    print(o)
    # date col 생성 -> index
    for i in range(raw_total): # 1500 -> 15*1 -1 , 30 = 15*2 -1 , 45, 60, 75,
        index_date["date"].append(o[i]["date"])  # 15, 30, 45
    print(index_date)

    # 각 site의 col 생성
    for site2 in count_row:
        count += 1
        count_cursor = db.result_table.find({"word": word, "site": site2},
                                            {"_id": 0, "count": 1, "date": 1}).sort([("date", 1)])

        o = OrderedDict()
        o = list(count_cursor) # 결과 가져옴.
        print("-----------")
        print(o)

        cur_sum = 0
        # last 개의 row 생성
        for k in range(raw_total): # 0,1,2,3,4 => last
            # 5. 15일간 누적치 : o[k]["count"]
            # 6. 이전 15일치에 다음 15일치 누적
            cur_sum += o[k]["count"]
            # 7. row-i, col-'site'의 데이터를 count_row col-'counts'에 추가
            count_row[site2].append(cur_sum)
        print(str(count)+" : " + site2 + " : " + str(o[k]["date"]) + " : " + str(cur_sum))
    write_csv(word, count_row, index_date)
    index_date["date"].clear()


def write_csv(word, row, index):
    # row[0:14]를 df에 추가함
    result_df = pd.DataFrame(row, index=index["date"])
    print(result_df)

    # 4. word_sum.csv에다가 저장
    if os.path.isfile("./sum/" + word + "_sum.csv"):
        print("--------is dir--------")
        # 이미 있는 word_sum.csv 파일에 추가할때
        result_df.to_csv("./sum/" + word + "_sum.csv", mode='a', header=False)
    else:  # 처음 만드는 사이트_data.csv 파일에 추가할때
        # 저장할때, 명시적으로 index를 설정하지 않거나, index=False 안해주면 "Unnamed: 0" 생성됨
        if word == "대깨*":
            result_df.to_csv("./sum/대깨-_sum.csv")
            return
        if word == "아시겠어요?":
            result_df.to_csv("./sum/아시겠어요-_sum.csv")
            return
        result_df.to_csv("./sum/" + word + "_sum.csv") # word _normalized


def test_main():
    try:
        #DB연결
        uri = "mongodb://%s:%s@%s" % (quote_plus("admin123"), quote_plus("1234"), "wolfwatch.dlinkddns.com:27017/admin")
        client = MongoClient(uri)
        db = client.crawler_db

        #sum값 추출할 검색어 이름 리스트
        #
        name_list = ["어나더레벨", "근손실", "창조주", "대깨*", "대깨문","연서복", "킹리적", "씹덕아", "어림도 없지", "피셜", "자낳괴", "갑분싸", "나락", "뇌절",
                    "스밍", "믿으셔야", "알못", "문찐", "믿거페", "JMT", "자만추", "애빼시",
                     "만찢", "시강", "팬아저", "일코노미", "법블레스유", "아시겠어요?"]

        for name in name_list:
            print(name)
            make_cursum(db, name)

    except:
        traceback.print_exc()


test_main()
