import glob
import json
from copy import deepcopy

from utils.benchmark_result import BaselineResult
from utils.data_utils import read_datys_data
from facos.DATYS.datys import infer_get_top_candidates

def benchmark_datys():
    data = read_datys_data()
    predictions  = {}
    all_baseline_result = []

    with open('data/api_method_candidates.json', "r") as fp:
        api_method_candidates = json.load(fp)
    with open(f"data/test_threads.json", "r") as fp:
        test_threads = json.load(fp)
    with open(f"data/apis_having_emb_test_set.json", "r") as fp:
        apis = json.load(fp)

    test_dict = {}
    with open(f"data/text_code_pairs_test.jsonl", "r") as fp:
        for line in fp.readlines():
            stack_pairs = json.loads(line)
            _idx = stack_pairs['idx']
            thread_id = stack_pairs['thread_id']
            simple_name = stack_pairs['simple_name']
            target_fqn = stack_pairs['target_fqn']
            cls_label = stack_pairs['cls_label']
            if target_fqn not in test_dict:
                test_dict[target_fqn] = {}
            if thread_id not in test_dict[target_fqn]:
                test_dict[target_fqn][thread_id]= []
            if _idx not in test_dict[target_fqn][thread_id]:
                test_dict[target_fqn][thread_id].append(_idx)

    
    with open("data/search_thread_from_es.json", "r") as fp:
        queried_threads = json.load(fp)
    with open("data/title.json", "r") as fp:
        thread_titles = json.load(fp)
    with open("data/tags.json", "r") as fp:
        thread_tags = json.load(fp)

    for api in apis:
        fqn = api
        if api not in data:
            continue
        if api not in test_dict:
            continue
        labels_ = data[api]
        labels_ = [lbl for lbl in labels_ if lbl in test_threads]

        simple_name = fqn.split(".")[-1]

        threads = queried_threads[api]
        
        if fqn not in predictions:
            predictions[fqn] = []

        for thread_id in threads:
            thread_title = thread_titles[thread_id]
            thread_tag = ";".join(thread_tags[thread_id])
            cand_dict = {}

            cand_dict = api_method_candidates[simple_name]
            thread_path = glob.glob("data/so_threads/"+str(thread_id)+".*")[0]

            with open(thread_path, "r") as tfp:
                thread_content = tfp.read()
            
            list_mentions = infer_get_top_candidates(thread_id, thread_content, thread_title, thread_tag, simple_name, cand_dict)
            predicted_apis = list_mentions[0]['preds']
            if fqn in predicted_apis:
                predictions[fqn].append(thread_id)

        benchmark_result = BaselineResult(fqn, labels_, predictions[fqn])
        all_baseline_result.append(benchmark_result)
        

    Prec = sum([result.prec for result in all_baseline_result])/len(all_baseline_result)
    Recall = sum([result.recall for result in all_baseline_result])/len(all_baseline_result)
    F1 = sum([result.f1 for result in all_baseline_result])/len(all_baseline_result)

    print(f"Nrof APIs: {len(all_baseline_result)}")
    print("avg Prec: ", Prec)
    print("avg Recall: ", Recall)
    print("avg F1: ", F1)


if __name__ == "__main__":
    benchmark_datys()