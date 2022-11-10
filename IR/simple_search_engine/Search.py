'''
开发时间：2021/12/12 19:29

python 3.8.5

开发人；Lasseford Wang

'''


import json
from elasticsearch import Elasticsearch
import urllib.request

INDEX_NAME = 'searchengine'


es = Elasticsearch()


def simpleSearch(input):
    query = {
        "query": {
            "multi_match": {
                "query": input,
                "fields": ["name^3", "othername^3","subtitleMade","type","commentsN","episodes","downloadN","state","downloadNweek"]
            }
        },
        "from":0,
        "size":10,
        "sort":[],
        "aggs":{}
    }
    res = es.search(index=INDEX_NAME, body=query)
    return res


def dateSearch(time1,time2):
    query = {"query":{"bool":{"must":[{"range":{"updatetime":{"gt":time1,"lt":time2}}}],"must_not":[],"should":[]}},"from":0,"size":10,"sort":[],"aggs":{}}
    res = es.search(index=INDEX_NAME,body=query)
    return res


def OnsiteSearch(url,input):
    query = {"query":{"bool":{"must":[{"match":{"type":url[-1],}},{"match":{"name":input}}],"must_not":[],"should":[]}},"from":0,"size":10,"sort":[],"aggs":{}}
    res = es.search(index=INDEX_NAME, body=query)
    return res


def phraseSearch(input):
    query={
        "query": {
            "multi_match": {
                "query": input,
                "fields": ["name","othername"],
                "type": "phrase"
            }
        },
        "from":0,
        "size":10,
        "sort":[],
        "aggs":{}
    }
    res = es.search(index=INDEX_NAME, body=query)
    return res

def wildcardSearch(input):
    s=input+'*'
    query ={
        "query": {
            "bool":{
                "should":[
                    {
                        "wildcard":{
                            "name":s
                        }
                    },
                    {
                        "wildcard":{
                            "othername":s
                        }
                    }
                ]
            }
        },
        "from":0,
        "size":10,
        "sort":[],
        "aggs":{}
    }
    res = es.search(index=INDEX_NAME, body=query)
    return res


def printres(res):
    if(res['hits']['hits']):
        s = ''
        for i in res['hits']['hits']:
            for itm in list(i['_source'].values()):
                s = s + str(itm) + '\n'
        return s
    else:
        return "No search result !"


def getHtml(url):
    html = urllib.request.urlopen(url).read()
    return html


def saveHtml(file_name, file_content):
    #    注意windows文件命名的禁用符，比如 /
    file_name="./htmlStore/"+file_name.replace('/', '_')
    with open(file_name + ".html", "wb") as f:
        #   写文件用bytes而不是str，所以要转码
        f.write(file_content)


def storeHtml(url):
    saveHtml(url,getHtml(url))
