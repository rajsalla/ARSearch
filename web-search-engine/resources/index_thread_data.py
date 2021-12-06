import os
import json
# from elasticsearch import Elasticsearch
from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import Index, Search, Mapping, DocType

# es_client = Elasticsearch([{'host': os.getenv("HOST"), 'port': os.getenv("PORT")}], http_auth=http_auth)

def main():
    with open("threads_to_index.json", "r") as fp:
        data = json.load(fp)
    
    hosts = [os.getenv("HOST")]
    http_auth = (os.getenv("USERNAME"), os.getenv("PASSWORD"))
    port = os.getenv("PORT")
    client = connections.create_connection(hosts=hosts, http_auth=http_auth, port=port)
    INDEX_NAME = "so_related_threads"
    DOC_TYPE = "api"
    
    for fqn, related_threads in data.items():
        doc_body = {
            "threads": related_threads
        }
        client.index(index=INDEX_NAME, doc_type=DOC_TYPE, id=fqn, body=doc_body)



if __name__ == "__main__":
    main()