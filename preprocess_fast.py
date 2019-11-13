import csv
import os
from collections import OrderedDict
from time import sleep

import pandas as pd
from pymongo import MongoClient
from urllib.parse import quote_plus
import traceback
from datetime import datetime, timedelta


def make_cursum(db, word):
    index_date = OrderedDict()
    count_row = OrderedDict()

    # 1. for site in siteList :
    #   find ((word) && count > 0)에 대해서 sort(date) 찾은다음
    zero_cursor = db.result_table.find({"word": word, "count": {"$gt": 0}}, {"_id": 0}).sort([("date", 1)])
    # 2. 첫날(=zeroDay)을 찾음
    zero_day = zero_cursor[0]["date"]
    print(zero_cursor[0])

    # 6. 빠른 날짜 부터
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
    list_cursor = db.result_table.find({"word": word, "site": site_list[0], "date": {"$gte": zero_day}}
                                       , {"_id": 0, "date": 1, "count": 1}).sort([("date", 1)])
    raw_total = list_cursor.count()
    last = int(raw_total / sum_day)
    last2 = int(raw_total % sum_day)
    total_count = raw_total - last2 # 1500

    o = OrderedDict()
    o = list(list_cursor)
    print(o)
    # date col 생성 -> index
    for k in range(last): # 1500 -> 15*1 -1 , 30 = 15*2 -1 , 45, 60, 75,
        index_date["date"].append(o[(sum_day*(k+1))-1]["date"])  # 15, 30, 45
    print(index_date)

    # 각 site의 col 생성
    for site2 in count_row:
        count += 1
        count_cursor = db.result_table.find({"word": word, "site": site2, "date": {"$gte": zero_day}},
                                            {"_id": 0, "count": 1, "date": 1}).sort([("date", 1)])

        o = OrderedDict()
        o = list(count_cursor) # 결과 가져옴.
        print("-----------")
        print(o)

        cur_sum = 0
        # last 개의 row 생성
        for k in range(last): # 0,1,2,3,4 => last
            # 5. 15일간 누적치를 구함
            cur = 0
            for i in range(sum_day): # 15 개씩 누적
                cur += o[cnt+i]["count"] # 0~14, 15~ 29, 30~44
            # 6. 이전 15일치에 다음 15일치 누적
            cur_sum += cur
            cnt += sum_day
            # 7. row-i, col-'site'의 데이터를 count_row col-'counts'에 추가
            count_row[site2].append(cur_sum)
            print(o[cnt-1]["date"])
        print(str(count)+" : " + site2 + " : " + str(o[cnt-1]["date"]) + " : " + str(cur_sum))
        cnt = 0
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
        result_df.to_csv("./sum/" + word + "_sum.csv")


def test_main():
    try:
        #DB연결
        uri = "mongodb://%s:%s@%s" % (quote_plus("admin123"), quote_plus("1234"), "wolfwatch.dlinkddns.com:27017/admin")
        client = MongoClient(uri)
        db = client.crawler_db
        site_list = OrderedDict()

        #sum값 추출할 검색어 이름 리스트
        name_list = [ "어나더레벨"]

        for name in name_list:
            print(name)
            make_cursum(db, name)

    except:
        traceback.print_exc()


test_main()
