import traceback
import json
import re



from .utils.code_extractor import *
from .Thread import Thread, ThreadInfer
from .Metrics import eval_mentions


data_labeling_dir = "/app/data/so_threads/"
tags_dir = "data/tags.json"
title_dir = "data/title.json"
api_method_candidates_dir = "data/tags.json"

with open(tags_dir, "r") as fp:
    tags_dict = json.load(fp)
with open(title_dir, "r") as fp:
    title_dict = json.load(fp)
with open(api_method_candidates_dir, "r") as fp:
    api_cands = json.load(fp)
        
from utils.datys_utils import tokenize


def type_scoping_infer_top_candidates_plus(mention, thread, candidates, return_api_term_count=False):
    fn_name = mention['name'].split(".")[-1]
    fn_name_caller = ".".join(mention['name'].split(".")[:-1])
    tags = thread.tags
    text_scope = thread.get_title() + " " +thread.get_text_wo_label() + " ".join(tags)
    code_snippets = thread.get_code()
    text_scope_tokens, content_vocabs = tokenize(text_scope)
    try:
        code_snippets_tokens, _ =tokenize(code_snippets)
    except Exception as e:
        traceback.print_exc()

        print(code_snippets)
        print(thread.thread_id)
        exit()

    text_tokens =re.findall("\w+(?:'\w+)?|[^\w\s]",(re.sub("\/\/(.*?)\\n", " ", text_scope)).replace("\n", ' '))
    code_tokens =re.findall("\w+(?:'\w+)?|[^\w\s]",(re.sub("\/\/(.*?)\\n", " ", code_snippets)).replace("\n", ' '))
    
    filtered_cands = []
    
    for fqn, lib in candidates.items():
        for tag in tags:
            if tag in lib:
                filtered_cands.append(fqn)
                break
    candidates = filtered_cands
    org_cands = deepcopy(candidates)

    score_dict = {} 
    has_type_one = {}
    can_component_scopes = {}
    for can_i, can in enumerate(candidates): 
        score_dict[can] = 0
        can_component_scopes[can] = []
        has_type_one[can] = False
        for pos_type in mention['p_types']:
            can_parts = can.split(".")
            can_class =can_parts[-2]
            if pos_type in can:
                if len(pos_type.split(".")) > 1:
                    if pos_type.split(".")[-1] != fn_name:
                        focus_str = pos_type.split(".")[-2]
                    else:
                        focus_str = pos_type.split(".")[-1]
                    
                else:
                    focus_str = pos_type.split(".")[-1]

                if focus_str == can_class:
                    score_dict[can] += 1
                    can_component_scopes[can].append("mention")

    for can_i, can in enumerate(candidates):
        class_name = can.split(".")[-2]
        if fn_name_caller != "":
            if fn_name_caller == class_name:
                score_dict[can] += 1
                can_component_scopes[can].append("code")

        if class_name in text_scope_tokens:
            score_dict[can] += 1
            can_component_scopes[can].append("text")
        
        if class_name in code_snippets_tokens:
            score_dict[can] += 1
            

    score_list = [(api, score) for api, score in score_dict.items()]
    score_list = sorted(score_list, key=lambda api: api[1], reverse=True)
    if len(score_list) == 0 or score_list[0][1] == 0:
        prediction = [can for can in org_cands]
        score = [0 for _ in org_cands]
        
    else:
        prediction = [api_score[0] for api_score in score_list]
        score = [api_score[1] for api_score in score_list]
    if len(prediction) < 2:
        for can in org_cands:
            if can not in prediction:
                prediction.append(can)
                score.append(0)
    terms = []
    for can in prediction:
        relevant_terms = 0
        can_terms = can.split(".")[:-1]
        for term in can_terms:
            if term in ["org", "collect", "google", "common", "com"]:
                continue
            if (term in text_scope_tokens) or (term in code_snippets_tokens):
                # print(term)
                relevant_terms += 1
        terms.append(relevant_terms)

    if not return_api_term_count:
        return prediction, score, [], can_component_scopes
    else:
        return prediction, score, [], can_component_scopes, terms

def infer_get_top_candidates_plus(thread_id, thread_content, thread_title, thread_tags, simple_method_name, api_candidates, scale=0.7):
    list_all_predicted_mentions = []
    a_thread = ThreadInfer(thread_id, thread_content, thread_title, thread_tags)
    possible_type_list = a_thread.get_possible_type_dict()
    
    p_type_dict = a_thread.extract_possible_types()
    
    text_mentions = a_thread.get_api_mention_text_and_code()
    text_mentions = [mention for mention in text_mentions if mention['name'].split(".")[-1] == simple_method_name]
    
    text_mentions = [dict(t) for t in {tuple(d.items()) for d in text_mentions}]

    text_mentions = [{'name': simple_method_name,'thread_id': thread_id}]
    if simple_method_name == "weakKeys" and thread_id == "13379071":
        pass

    new_text_mentions = []
    for m_idx, m in enumerate(text_mentions):
        mention = deepcopy(m)
        mention['p_types'] = []
        list_p_types = []
        simple_m_name = m['name'].split(".")[-1]
        prefix = ".".join(m['name'].split(".")[:-1])
        if prefix != "":
            if prefix in p_type_dict:
                p_types_of_prefix = p_type_dict[prefix]
                for p_type in p_types_of_prefix:
                    list_p_types.append(p_type)
                    
        elif m['name'] in p_type_dict:
            method_related_p_types = p_type_dict[m['name']]
            for p_type in method_related_p_types:
                list_p_types.append(p_type)
                if p_type in p_type_dict:
                    list_p_types += p_type_dict[p_type]
        mention['p_types'] = list(set(list_p_types))
        mention['thread'] = a_thread.thread_id
        new_text_mentions.append(mention)


    datys_cand_term_counts = []
    datys_cand_max_scores = []
    datys_cands = []
    for mention in new_text_mentions:
        mention['preds'], mention['score'], _, mention['candidates_scope'], mention['term_counts'] = type_scoping_infer_top_candidates_plus(mention, a_thread, api_candidates, return_api_term_count=True)

        for pred_i, pred in enumerate(mention['preds']):
            if pred not in datys_cands:
                datys_cands.append(pred)
                datys_cand_max_scores.append(0)
                datys_cand_term_counts.append(mention['term_counts'][pred_i])
            pred_pos_id = datys_cands.index(pred)
            pred_score = mention['score'][pred_i]
            if pred_score > datys_cand_max_scores[pred_pos_id]:
                datys_cand_max_scores[pred_pos_id] = pred_score
            
    
    can_score_term_non_zero = [(can, score, term_count) for (can, score, term_count) in list(zip(datys_cands, datys_cand_max_scores, datys_cand_term_counts)) if term_count != 0]
    if len(can_score_term_non_zero)!= 0:
        datys_cands, datys_cand_max_scores, datys_cand_term_counts = list(map(list, zip(*can_score_term_non_zero)))
    else:
        list_all_predicted_mentions = [{'preds':[], 'score':[], 'term_counts': [], 'thread_id': thread_id}]
        return list_all_predicted_mentions

    score_min = 0
    score_max = max(datys_cand_max_scores)
    list_all_predicted_mentions = [{'preds':[], 'score':[], 'term_counts': [], 'thread_id': thread_id}]
    
    
    scaled_scores = [scale*(score-score_min)/(score_max-score_min) if score_max!=score_min else 0 for score in datys_cand_max_scores]

    for cand_score_term in list(zip(datys_cands, scaled_scores, datys_cand_term_counts)):
        cand, score, term_count = cand_score_term
        list_all_predicted_mentions[0]['preds'].append(cand)
        list_all_predicted_mentions[0]['score'].append(score)
        list_all_predicted_mentions[0]['term_counts'].append(term_count)

    return list_all_predicted_mentions