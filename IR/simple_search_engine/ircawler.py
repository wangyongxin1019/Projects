'''
开发时间：2021/12/12 14:43

python 3.8.5

开发人；Lasseford Wang

'''

import requests
from requests.exceptions import RequestException
import re
import json
import traceback

from elasticsearch import Elasticsearch

from datetime import datetime

INDEX_NAME = 'searchengine'


es = Elasticsearch()

def get_one_page(url,headers):
   try:
        response = requests.get(url,headers = headers)
        response.encoding = 'utf-8'
        if response.status_code == 200:
            return response.text
        return None
   except RequestException:
       return None

def parse_one_page(html):
    p = re.compile('<title>.*?</title>',re.S)
    Item = re.findall(p,html)

    p=re.compile("</td><td class='val' colspan='3'>.*?</td>",re.S)
    Item2=re.findall(p,html)


    pattern = re.compile("</td><td class='val'>.*?</td>",re.S)
    items = re.findall(pattern, html)
    if(len(items)==7):
        yield {
                'name':str(Item).replace('<title>','').replace('</title>',''),
                'othername':str(Item2).replace("</td><td class='val' colspan='3'>",'').replace('</td>',''),
                'subtitleMade': str(items[0]).replace("</td><td class='val'>",'').replace('</td>',''),

                'type': str(items[1]).replace("</td><td class='val'>",'').replace('</td>',''),
                'updatetime': str(items[2]).replace("</td><td class='val'>",'').replace('</td>',''),

                'commentsN': str(items[3]).replace("</td><td class='val'>",'').replace('</td>',''),
                'episodes': str(items[4]).replace("</td><td class='val'>",'').replace('</td>',''),
                'downloadN':str(items[5]).replace("</td><td class='val'>",'').replace('</td>',''),
                'state':str(items[6]).replace("</td><td class='val'>",'').replace('</td>',''),
                'downloadNweek':str(items[7]).replace("</td><td class='val'>",'').replace('</td>','')
        }
    else:
        return


def create_index():
    if es.indices.exists(INDEX_NAME):
        es.indices.delete(INDEX_NAME)
        print('删除存在的索引 \'{}\' ，并创建一个新的索引'.format(INDEX_NAME))
    result = es.indices.create(index=INDEX_NAME, ignore=400)
    print(result)
    put_map()
    return


def put_map():
    es.indices.put_mapping(
        body=
        {
            "dynamic": "strict",
            #"_source": {"enabled": "false"},
            "properties":
                {
                    "url":{"type": "keyword"}
                    ,"name": {"type": "keyword"}
                    ,"othername": {"type": "keyword"}
                    ,"subtitleMade": {"type": "keyword"}
                    ,"type": {"type": "keyword"}
                    ,"updatetime": {"type": "date","format":"yyyy-MM-dd HH:mm:ss"}
                    ,"commentsN": {"type": "keyword"}
                    ,"episodes": {"type": "keyword"}
                    ,"downloadN": {"type": "keyword"}
                    ,"state": {"type": "keyword"}
                    ,"downloadNweek": {"type": "keyword"}
                }
        }
        ,index=INDEX_NAME
        ,doc_type='12club'
        ,include_type_name=True
    )


def index_to_es(url,html):
    print("start!!")
    p = re.compile('<title>.*?</title>', re.S)
    Item = re.findall(p, html)

    p = re.compile("</td><td class='val' colspan='3'>.*?</td>", re.S)
    Item2 = re.findall(p, html)

    pattern = re.compile("</td><td class='val'>.*?</td>", re.S)
    items = re.findall(pattern, html)
    print(items)
    if (len(items) == 8):
        josndoc = {
            'url':url,
            'name': str(Item[0]).replace('<title>', '').replace('</title>', '').replace(' - 动画 - 十二社区','').replace(' - 漫画 - 十二社区','').replace(' - 音乐 - 十二社区','').replace(' - 游戏 - 十二社区','').replace(' - 轻小说 - 十二社区','').replace(' - 视频 - 十二社区','').replace('[完]',''),
            'othername': str(Item2[0]).replace("</td><td class='val' colspan='3'>", '').replace('</td>', ''),
            'subtitleMade': str(items[0]).replace("</td><td class='val'>", '').replace('</td>', ''),

            'type': str(items[1]).replace("</td><td class='val'>", '').replace('</td>', ''),
            'updatetime': str(items[2]).replace("</td><td class='val'>", '').replace('</td>', ''),

            'commentsN': str(items[3]).replace("</td><td class='val'>", '').replace('</td>', ''),
            'episodes': str(items[4]).replace("</td><td class='val'>", '').replace('</td>', ''),
            'downloadN': str(items[5]).replace("</td><td class='val'>", '').replace('</td>', ''),
            'state': str(items[6]).replace("</td><td class='val'>", '').replace('</td>', ''),
            'downloadNweek': str(items[7]).replace("</td><td class='val'>", '').replace('</td>', '')
        }
        print(josndoc)
        try:
            es.index(index=INDEX_NAME, doc_type="12club", body=josndoc)
            print("Done indexing the doc")
        except Exception as ex:
            traceback.print_exc()
            print("Failed to index the document {}".format(jsonMapDoc))
    else:
        pass

def splitStr(s):
    if ' - 动画 - 十二社区' in s:
        s.replace(' - 动画 - 十二社区','')
    if ' - 漫画 - 十二社区' in s:
        s.replace(' - 漫画 - 十二社区','')
    if ' - 音乐 - 十二社区' in s:
        s.replace(' - 音乐 - 十二社区','')
    if ' - 游戏 - 十二社区' in s:
        s.replace(' - 游戏 - 十二社区','')
    if ' - 轻小说 - 十二社区' in s:
        s.replace(' - 轻小说 - 十二社区','')
    if ' - 视频 - 十二社区' in s:
        s.replace(' - 视频 - 十二社区','')
    return s


def main():
    create_index()

    for i in range(1,3000):
        url = 'http://12club.nankai.edu.cn/programs/'+str(i)
        print(url)
        user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
        headers = {'User-Agent': user_agent}
        html = get_one_page(url, headers)
        index_to_es(url,html)


if __name__ == '__main__':
    main()
