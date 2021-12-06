import os
import json
import glob
# from elasticsearch import Elasticsearch
from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import Index, Search, Mapping, DocType
from thread_reader import TextCodeThreadReader

# es_client = Elasticsearch([{'host': os.getenv("HOST"), 'port': os.getenv("PORT")}], http_auth=http_auth)

def main():
    # with open("threads_to_index.json", "r") as fp:
    #     data = json.load(fp)
    
    hosts = [os.getenv("HOST")]
    http_auth = (os.getenv("USERNAME"), os.getenv("PASSWORD"))
    port = os.getenv("PORT")
    client = connections.create_connection(hosts=hosts, http_auth=http_auth, port=port)
    INDEX_NAME = "thread_contents"
    DOC_TYPE = "thread"
    
    # for fqn, related_threads in data.items():
    thread_paths = glob.glob("/facos_data/so_threads/*")
    with open("/facos_data/tags.json", "r") as tag_fp:
        tag_dict = json.load(tag_fp)
    
    with open("/facos_data/title.json", "r") as title_fp:
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
        # print(thread_id)
        # for text_i ,text in enumerate(doc_body['texts']):
        #     print(text_i)
        #     print(text)

        # print("\n=====================\n")
        # for code_i, code in enumerate(doc_body['codes']):
        #     print(code_i)
        #     print(code)
        # exit()
        client.index(index=INDEX_NAME, doc_type=DOC_TYPE, id=thread_id, body=doc_body)



if __name__ == "__main__":
    main()