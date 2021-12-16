import os
import json
# from elasticsearch import Elasticsearch
from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import Index, Search, Mapping, DocType


def main():
    with open("../data/threads_to_index.json", "r") as fp:
        data = json.load(fp)
    
    hosts = [os.getenv("HOST")]
    port = os.getenv("PORT")
    client = connections.create_connection(hosts=hosts, port=port)

    INDEX_NAME = "so_related_threads_clone"
    DOC_TYPE = "api"
    
    for fqn, related_threads in data.items():
        doc_body = {
            "threads": related_threads
        }
        client.index(index=INDEX_NAME, doc_type=DOC_TYPE, id=fqn, body=doc_body)



if __name__ == "__main__":
    main()