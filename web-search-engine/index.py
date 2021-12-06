#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Anthony Sigogne"
__copyright__ = "Copyright 2017, Byprog"
__email__ = "anthony@byprog.com"
__license__ = "MIT"
__version__ = "1.0"

__modified_by__ = "Kien Luong"

import re
import os
# import url
# import crawler
import requests
import json
# import query
import traceback
from copy import deepcopy
from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch


from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import Index, Search, Mapping, DocType
from language import languages

# init flask app and import helper
app = Flask(__name__)
with app.app_context():
    from helper import *

# initiate the elasticsearch connection
hosts = [os.getenv("HOST")]
http_auth = (os.getenv("USERNAME"), os.getenv("PASSWORD"))
port = os.getenv("PORT")
client = connections.create_connection(hosts=hosts, http_auth=http_auth, port=port)


with open("resources/github_files_by_api.json", "r") as gh_fp:
    gh_files_by_api = json.load(gh_fp)

with open("resources/code_link_dict.json", "r") as code_link_fp:
    code_link_dicts = json.load(code_link_fp)
    
@app.route("/index", methods=['POST'])
def index():
    """
    URL : /index
    Index a new URL in search engine.
    Method : POST
    Form data :
        - url : the url to index [string, required]
    Return a success message.
    """
    # get POST data
    data = dict((key, request.form.get(key)) for key in request.form.keys())
    if "url" not in data :
        raise InvalidUsage('No url specified in POST data.')

    # launch exploration job
    # index_job.delay(data["url"])

    return "Indexing started"

@app.route("/search", methods=['POST'])
def search():
    """
    URL : /search
    Query engine to find a list of relevant URLs.
    Method : POST
    Form data :
        - query : the search query [string, required]
        - hits : the number of hits returned by query [integer, optional, default:10]
        - start : the start of hits [integer, optional, default:0]
    Return a sublist of matching URLs sorted by relevance, and the total of matching URLs.
    """
    def format_result(hit, highlight) :
        # highlight title and description
        title = hit["title"]
        description = hit["description"]
        if highlight :
            if "description" in highlight :
                description = highlight["description"][0]+"..."
            elif "body" in highlight :
                description = highlight["body"][0]+"..."
            """if "title" in highlight :
                title = highlight["title"][0]"""

        # create false title and description for better user experience
        if not title :
            title = hit["domain"]
        if not description :
            description = "..."

        return {
            "title":title,
            "description":description,
            "url":hit["url"],
            "thumbnail":hit.get("thumbnail", None)
        }

    # get POST data
    data = dict((key, request.form.get(key)) for key in request.form.keys())
    if "query" not in data :
        raise InvalidUsage('No query specified in POST data')
    start = int(data.get("start", "0"))
    hits = int(data.get("hits", "10"))
    if start < 0 or hits < 0 :
        raise InvalidUsage('Start or hits cannot be negative numbers')

    SO_API_RELATED_RES_INDEX_NAME = "so_related_threads"
    SO_THREAD_CONTENT_INDEX_NAME = "thread_contents"

    
    # analyze user query
    try:
        querying_id = data["query"] # querying_id contains the fqn
        if querying_id[-1] == '+':
            querying_id = querying_id[:-1]
        
        simple_name = querying_id.split(".")[-1]

        # get github files:
        if querying_id in gh_files_by_api:
            gh_results = deepcopy(gh_files_by_api[querying_id])
        else:
            gh_results = []
        
        for gh_i, gh_result in enumerate(gh_results):
            highlighted_index = gh_result['line_number']+1 - gh_result['start_line']
            gh_results[gh_i]['go_to_top'] = False
            codes = ""
            for i, line_no in enumerate(list(range(gh_result['start_line'], gh_result['end_line']+1))):
                
                if i == highlighted_index:
                    if gh_result['code'][i].strip().split()[0].lower() not in ["protected", "public", "private"]:
                        gh_results[gh_i]['go_to_top'] = True
                    codes += "<span class='invocation'>" + str(line_no) + "\t" + gh_result['code'][i] +"</span>" 
                else:
                    codes += str(line_no) + "\t" +gh_result['code'][i]

            gh_result['link'] = gh_result['link'][gh_result['link'].find('http'):]+ "#L" +str(gh_result['line_number']+1)
            gh_results[gh_i]['code'] = codes

        top_results = []
        bot_results = []
        for gh_result in gh_results:
            if gh_result['go_to_top']:
                top_results.append(gh_result)
            else:
                bot_results.append(gh_result)
        gh_results = top_results + bot_results
        gh_results = gh_results[:10]

        
            
        api_related_threads_result = client.get(id=querying_id, index=SO_API_RELATED_RES_INDEX_NAME, doc_type="api")
        threads = api_related_threads_result['_source']['threads']
        total = len(threads)
        for thread_i, thread in enumerate(threads):
            thread_content = client.get(id=thread['thread_id'], index=SO_THREAD_CONTENT_INDEX_NAME, doc_type="thread")["_source"]
            if 'shown_text' not in threads[thread_i]:
                for code in thread_content['codes']:
                    if simple_name in code:
                        simple_name_pos_start = code.find(simple_name)
                        simple_name_pos_end = simple_name_pos_start +len(simple_name)
                        code = code[:simple_name_pos_start] + "<span class='api-mention'>" + simple_name + "</span>" + code[simple_name_pos_end:]
                        threads[thread_i]['shown_text'] = code
            for text in thread_content['texts']:
                if simple_name in text:
                    text_temp = deepcopy(text)
                    text_return = ""
                    while simple_name in text_temp:
                        simple_name_pos_start = text_temp.find(simple_name)
                        simple_name_pos_end = simple_name_pos_start +len(simple_name)
                        text_return += text_temp[:simple_name_pos_start] + "<span class='api-mention'>" + simple_name + "</span>"
                        text_temp = text_temp[simple_name_pos_end:]
                    text_return += text_temp
                    threads[thread_i]['shown_text'] = text_return
                    break
            

        if querying_id in code_link_dicts:
            code_link_dict = code_link_dicts[querying_id]
            for thread_i, thread in enumerate(threads):
                code_indices = code_link_dict[str(thread_i)]
                threads[thread_i]['linked_gh_code'] = [gh_results[int(i)-1] for i in code_indices]
        return jsonify(total=total, results=threads, gh_results=gh_results)
        
    except Exception as e:
        traceback.print_exc()
        total = 0
        results = []
        return jsonify(total=total, results=results, gh_results=gh_results)