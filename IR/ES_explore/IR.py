'''
开发时间：2021/11/17 16:58

python 3.8.5

开发人；Lasseford Wang

'''

import os
import sys
from email.parser import Parser
from elasticsearch import Elasticsearch
from datetime import date
import traceback
import re

INDEX_NAME = 'enron-email_2'


es = Elasticsearch()


p = Parser()

def create_index():
    if es.indices.exists(INDEX_NAME):
        es.indices.delete(INDEX_NAME)
        print('删除存在的索引 \'{}\' ，并创建一个新的索引'.format(INDEX_NAME))
    result = es.indices.create(index=INDEX_NAME, ignore=400)
    print(result)
    put_map()
    return


def get_enron_eml_content(eml_file_to_open):
    data_file = open(eml_file_to_open,encoding='gbk',errors='ignore')
    contents = ""
    try:
        for line in data_file:
            contents += line
    finally:
        data_file.close()
    return contents


def get_att(eml_file_to_open):
    s=" - "
    s=s+r'(.+?)\.'
    data_file = open(eml_file_to_open,encoding='gbk',errors='ignore')
    tt=data_file.readlines()
    lastline = tt[-1]
    m=re.findall(s,lastline)
    if len(m) != 0:
        return str(m)
    else:
        return " "



def put_map():
    es.indices.put_mapping(
        body=
        {
            "dynamic": "strict",
            #"_source": {"enabled": "false"},
            "properties":
                {
                    "content-transfer-encoding": {"type": "text"}
                    ,"message_body": {"type": "text"}
                    ,"content-type": {"type": "text"}
                    ,"x-bcc": {"type": "text"}
                    ,"from": {"type": "keyword"}
                    ,"x-from": {"type": "text"}
                    ,"x-filename": {"type": "text"}
                    ,"x-folder": {"type": "text"}
                    ,"to": {"type": "keyword"}
                    ,"x-to": {"type": "text"}
                    ,"mime-version": {"type": "keyword"}
                    ,"cc": {"type": "text"}
                    ,"x-cc": {"type": "text"}
                    ,"bcc": {"type": "text"}
                    ,"x-bcc": {"type": "text"}
                    ,"subject": {"type": "text"}
                    ,"message-id": {"type": "keyword"}
                    ,"x-origin": {"type": "text"}
                    #,"date": {"type": "date", "format": "EEE, dd MMM yyyy HH:mm:ss Z (z)"}
                    , "date": {"type": "keyword"}
                    ,"att":{"type":"keyword"}
                }
        }
        ,index=INDEX_NAME
        ,doc_type='enron-type'
        ,include_type_name=True
    )


def index_into_elasticsearch(nameOfFileToOpen, filename, contents):
    msg = p.parsestr(contents)
    jsonMapDoc = {}

    headers = dict(msg._headers)
    for key, value in headers.items():
        key = key.lower()
        if not value.find(",") == -1 and key != "date" and key != "subject":
            value = value.split(",")
            jsonMapDoc[key] = value
        else:
            jsonMapDoc[key] = value

    jsonMapDoc["message_body"] = msg._payload
    jsonMapDoc["att"] = get_att(nameOfFileToOpen)
    file_size = os.path.getsize(nameOfFileToOpen)
    try:
        es.index(index=INDEX_NAME, doc_type="enron-type", body=jsonMapDoc)
        print("Done indexing the doc")
    except Exception as ex:
        traceback.print_exc()
        print("Failed to index the document {}".format(jsonMapDoc))
    return


import datetime

def data_read():
    mail_dir = 'D:/Data1/maildir'
    create_index()

    for root, dirs, files in os.walk(mail_dir, topdown="false"):

        for filename in files:
            # get the file contents
            nameOfFileToOpen = "{0}/{1}".format(root, filename)
            contents = get_enron_eml_content(nameOfFileToOpen)
            index_into_elasticsearch(nameOfFileToOpen, filename, contents)



if __name__ == "__main__":
    data_read()





