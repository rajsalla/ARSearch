curl -X PUT "localhost:9200/?pretty" -H 'Content-Type: application/json' -d'
{
    "settings": {
        "number_of_shards": 1,
        "analysis": {
            "char_filter": {
                "replace_dot_with_space": {
                    "type": "pattern_replace",
                    "pattern": "\\.",
                    "replacement": " "
                }
            },
            "analyzer": {
                "api_analyzer_v2": { 
                    "tokenizer": "keyword",
                    "filter": [
                        "allow_parenthes_filter"
                    ],
                    "char_filter": [
                        "replace_dot_with_space"
                    ]
                }
            },
            "filter": {
                "allow_parenthes_filter": {
                    "type": "word_delimiter",
                    "type_table": [ "_ => ALPHA"],
                    "split_on_numerics": false,
                    "split_on_case_change": false,
                    "stem_english_possessive": true
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "threads": {
                
            }
        }
    }
}
'
