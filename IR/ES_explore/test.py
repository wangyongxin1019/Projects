'''
开发时间：2021/11/18 22:34

python 3.8.5

开发人；Lasseford Wang

'''


from elasticsearch import Elasticsearch


es=Elasticsearch()

query = {
     "query":{
          "bool":{
               "must":[{
                    "prefix":{
                         "from":"susan"
                    }
               }
               ],
               "must_not":[],
               "should":[]
          }
     },
     "from":0,
     "size":10,
     "sort":[],
     "aggs":{}
}

res = es.search(index='enron-email_2',body=query)

print(res)

qq={
     "query": {
          "bool": {
               "must": [
                    {
                    "match": {
                    "att": "['DRAW2']"
                    }
               }
          ],
          "must_not": [ ],
          "should": [ ]
          }
     },
     "from": 0,
     "size": 10,
     "sort": [ ],
     "aggs": { }
}

res = es.search(index='enron-email_2',body=qq)

print(res)