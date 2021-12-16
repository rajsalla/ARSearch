import json

from .utils.code_extractor import *
from .Thread import Thread
from .Metrics import eval_mentions
from utils.datys_utils import tokenize

tags_dir = "data/tags.json"
title_dir = "data/title.json"
api_method_candidates_dir = "data/tags.json"

with open(tags_dir, "r") as fp:
    tags_dict = json.load(fp)
with open(title_dir, "r") as fp:
    title_dict = json.load(fp)
with open(api_method_candidates_dir, "r") as fp:
    api_cands = json.load(fp)
        

def get_text_scope(mentions, thread):
    text_scope = thread.get_text_wo_label()
    
    text_scope_lines = text_scope.split("\n")
    for mention in mentions:
        line = text_scope_lines[mention['line_i']]
        line = line.replace(mention['name'], " ", 1)
        text_scope_lines[mention['line_i']] = line
    return "\n".join(text_scope_lines)


def type_scoping_infer_top_candidates(mention, thread, candidates):
    fn_name = mention['name'].split(".")[-1]
    fn_name_caller = ".".join(mention['name'].split(".")[:-1])
    tags = thread.tags
    text_scope = thread.get_title() + " " +thread.get_text_wo_label() + " ".join(tags)
    code_snippets = thread.get_code()
    text_scope_tokens, content_vocabs = tokenize(text_scope)
    
    filtered_cands = []
    for fqn, lib in candidates.items():
        for tag in tags:
            if tag in lib:
                filtered_cands.append(fqn)
                break
    candidates = filtered_cands
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
        
    score_list = [(api, score) for api, score in score_dict.items()]
    score_list = sorted(score_list, key=lambda api: api[1], reverse=True)
    if len(score_list) == 0 or score_list[0][1] == 0:
        prediction = []
        score = 0
    else:
        prediction = [api_score[0] for api_score in score_list if api_score[1] == score_list[0][1]]
        score = score_list[0][1]
    
    if len(prediction) > 0:
        return prediction, score, can_component_scopes[score_list[0][0]], can_component_scopes
    else:
        return prediction, score, [], can_component_scopes


def infer_get_top_candidates(thread_id, thread_content, thread_title, thread_tags, simple_method_name, api_candidates):
    list_all_predicted_mentions = []
    a_thread = Thread(thread_id, thread_content, thread_title, thread_tags)
    possible_type_list = a_thread.get_possible_type_dict()
        
    p_type_dict = a_thread.extract_possible_types()
    text_mentions = [{'name': simple_method_name,'thread_id': thread_id}]

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


    for mention in new_text_mentions:
        mention['preds'], mention['score'], mention['scopes'], mention['candidates_scope'] = type_scoping_infer_top_candidates(mention, a_thread, api_candidates)

    list_all_predicted_mentions += new_text_mentions
    return list_all_predicted_mentions