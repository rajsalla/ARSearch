import os
import json
import glob
# from elasticsearch import Elasticsearch
from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import Index, Search, Mapping, DocType
from TextCodeThreadReader import TextCodeThreadReader


def main():
    hosts = [os.getenv("HOST")]
    port = os.getenv("PORT")
    client = connections.create_connection(hosts=hosts, port=port)

    INDEX_NAME = "thread_contents_clone"
    DOC_TYPE = "thread"
    
    thread_paths = glob.glob("../data/so_threads/*")
    with open("../data/tags.json", "r") as tag_fp:
        tag_dict = json.load(tag_fp)
    
    with open("../data/title.json", "r") as title_fp:
        title_dict = json.load(title_fp)
    for thread_path in thread_paths:
        with open(thread_path, "r") as tfp:
            thread_content = tfp.read()
        ext = thread_path.split(".")[-1]
        thread_id = thread_path.split(os.sep)[-1].split(".")[0]
        thread_reader = TextCodeThreadReader(thread_content, thread_id, title_dict, tag_dict, ext)
        doc_body = {
            'texts': thread_reader.texts,
            'codes': thread_reader.code_blocks
        }
        client.index(index=INDEX_NAME, doc_type=DOC_TYPE, id=thread_id, body=doc_body)



if __name__ == "__main__":
    main()