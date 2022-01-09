# ARSearch: Searching for API Related Resources from Stack Overflow and GitHub


## Run the web server and UI
```
cd web-search-engine
docker compose up -d
```

## Run FACOS to predict Stack Overflow Resource for APIs
```
docker exec -it facos bash
cd facos/
python process_with_facos.py
```

## Import data to Elasticsearch
```
docker exec -it facos bash
cd facos/scripts
python index_so_content.py
python index_thread_data.py
```
## Export/ Import data from/to Elasticsearch

```
docker exec -it facos bash

# Intall elasticdump
npm install elasticdump -g

# Export elasticsearch index
elasticdump --input=http://elasticsearch:9200/so_related_threads --output=so_related_threads_mapping.json

# Import elasticsearch index
elasticdump --input=so_related_threads_mapping.json --output=http://elasticsearch:9200/so_related_threads_copy
thread_contents
```




