import json
from collections import OrderedDict

import pandas as pd
import numpy as np
from urllib.parse import quote_plus
import os
from pymongo import MongoClient


# kmeans_statemap.py 에다 단어별 결과물을 받게끔 함
def main(word):
    # world_map을 업데이트
    with open('./d3/world_map3.json', encoding='utf-8') as map_file:
        data_map = json.load(map_file)

    print(data_map)
    # data_map['count'] = 0
    # data_map['links'] = []
    # for i in range(len(data_map['nodes'])-1):
    #     for j in range(i, len(data_map['nodes'])):
    #         l = {}
    #         l['source'] = data_map['nodes'][i]['id']
    #         l['target'] = data_map['nodes'][j]['id']
    #         l['value'] = 0.0
    #         data_map['links'].append(l)
    # print(data_map)
    # with open('world_map.json', 'w', encoding="utf-8") as fp:
    #     json.dump(data_map, fp, ensure_ascii=False, indent=4)
    # exit(1)
    # csv 읽기

    if not os.path.isfile("./seq2/" + word + "_seq.json"):
        print(word + " file not exitst")
        return
    with open('./seq2/'+word+'_seq.json', encoding='utf-8') as data_file:
        data_now =json.load(data_file)
        if len(data_now['nodes']) < 1:
            return
        elif len(data_now['nodes'][0]) < 1:
            return
        print(data_now['nodes'])
        # 1. 현재 단어와 연관된 노드들 목록
        # 2. 노드들을 이용해서, 예)1번~10번 노드가 있을때
        # 3. 1번과 관련된 weight값들을 넣는다. (source: 1번, target : n번), (source: n번, target : 1번)
        # 4. for 문을 이용해서 1번 : 2번~ 10번까지, 그 다음 2번 : 3번 ~ 10번...
        # 5. ['count'] 를 이용해서 값을  조정해준다. weight = weight*(count/count+1)
        # 6. ['count'] += 1
        start = 0
        for i in range(len(data_map['nodes'])):
            source = data_map['nodes'][i]['id']

            now_size = len(data_map['nodes']) - (i + 1)
            start += now_size
            for j in range(i, len(data_now['links'])):
                if data_now['links'][j]['source'] == source:
                    # source를 찾으면, weight 값을 가지고와서
                    weight = data_now['links'][j]['value']

                    for k in range(len(data_map['nodes'])):
                        # 현재 target이 map에서 몇번인지(k) 찾는다
                        if data_now['links'][j]['target'] == data_map['nodes'][k]['id']:
                            # map에서 source의 번호와, tartget의 번호 : j, k를 이용해서 weight를 업데이트할 위치를 찾는다.
                            # 11, 10, 9, 8, 7 -> 11, 10, 9 -> 31 2*12 = 24 -> 22
                            # 0, 12, 23, 33, 42, 50
                            # 12, 24, 36, 48
                            # + (k-i) (i==2, k == 3 )
                            position = start + (k-i)
                            print(data_now['links'][j])
                            print(data_map['nodes'][i])
                            print(position)
                            print(len(data_map['links']))
                            data_map['links'][position]['value'] += weight
                            print(data_map['links'][position])
                            break

                if data_now['links'][j]['target'] == source:
                    # target을 찾으면, weight 값을 가지고와서
                    weight = data_now['links'][j]['value']
                    # k 1->7 // 7 -> 1 => 1,7의 weight에 적용 -> j 번째의 source가 몇번(k)인지 찾는다,

                    for k in range(len(data_map['nodes'])):
                        # 현재 source가 map에서 몇번인지(k) 찾는다
                        if data_now['links'][j]['source'] == data_map['nodes'][k]['id']:

                            # map에서 source의 번호와, tartget의 번호 : j, k를 이용해서 weight를 업데이트할 위치를 찾는다.
                            position = start + (k-i)
                            print(len(data_map['links']))
                            data_map['links'][position]['value'] += weight
                            print(data_map['links'][position])
                            break

    with open('./d3/world_map3.json', 'w',encoding='utf-8') as map_file:
        count = data_map['count']
        if count != 0:
            for i in range(len(data_map["links"])):
                weight = data_map['links'][i]['value']*(count/(count+1))
                data_map['links'][i]['value'] = weight

        data_map['count'] += 1

        json.dump(data_map, map_file, ensure_ascii=False, indent=4)
        print(data_map)


uri = "mongodb://%s:%s@%s" % (quote_plus("admin123"), quote_plus("1234"), "wolfwatch.dlinkddns.com:27017/admin")
client = MongoClient(uri)
db = client.crawler_db
list_cursor = db.word_table.find({},{"_id": 0,"word": 1})
for item in list_cursor:
    word = item['word']
    print(word)
    main(word)

exit(2)