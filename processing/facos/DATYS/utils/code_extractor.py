import os
import sys
import re
import time
from copy import deepcopy
from pprint import PrettyPrinter as PP
from pprint import pformat
pp = PP(depth=2)
pprint = pp.pprint

"""
Note:
[^\s\{]*: none whitespace char except { brace
\s: all whitespace
\S: all non-whitespace
"""
chars_not_space = "\S+"
chars_not_brace = "[^\s\{]*"
chars_not_parentheses = "[^\s\(]*"

space_p = "[\ \t]+"
space_s = "[\ \t]*"
comment_block = "/**.***/"
comment_line = "\/\/.*"


import_stmt = "(import)[\ \t]+([^\{]*?);"

assign_stmt  = "[;\{^]\s*([^;\{]*=)\s*([^;]*)|^\s*([^;\{]*=)\s*([^;]*)"
init_stmt = "([A-Za-z0-9_\.]+)\s+([A-Za-z0-9_\.]+)\s*;"
fn_call_stmt = "(new\s)?([A-Za-z0-9_\.]+[(])" 

fn_call_stmt_1 = "(new\s+[^\);]+?\)\.)([A-Za-z0-9_\.]+[(])" 
fn_call_stmt_2 = "(new.*[)][)]\.)([A-Za-z0-9_\.]+[(])"

stmt_types = [import_stmt, init_stmt, assign_stmt, fn_call_stmt, fn_call_stmt_1, fn_call_stmt_2]


def add_dict(a_dict, key, value, stmt_i, exclude_list=["return"]):
    if value in exclude_list:
        return 
    if key in a_dict:
        if value == "":
            return
        a_dict[key][value] = stmt_i
    else:
        if value == "":
            a_dict[key] = {}
        else:
            a_dict[key] = {value: stmt_i}

def get_method_pos(constructor_new):
    start = constructor_new.count('(')
    end =  constructor_new.count(')')
    pos = abs(start-end)
    return 0-pos

def clean_method_name(method_name):
    if method_name[-1] == "(":
        return method_name[:-1]
    return method_name

def extract_code(code_blk):
    var_type_dict = {}
    fn_var_dict = {}
    import_dict = {}
    lines = code_blk.split("\n")
    lines = [line for line in lines if len(line) > 0]
    code = " ".join(lines)
    for stmt_i, stmt_type in enumerate(stmt_types):
        pattern = re.compile(stmt_type)
        matches = pattern.findall(code)
        if len(matches) == 0:
            continue
        if stmt_i == 0:
            for match in matches:
                _type, var = match
                var = var.split(" ")[-1]
                if var[-1] == ";":
                    var = var[:-1]
                add_dict(import_dict, var, _type, stmt_i)
                
        if stmt_i == 1: # init
            for match in matches:
                _type, var = match
                if _type in ["import", "package", "static"]:
                    continue
                add_dict(var_type_dict, var, _type, stmt_i)

        if stmt_i == 2: # assign
            for match in matches:
                if match[0] == '':
                    var_and_type = match[2]
                else:
                    var_and_type = match[0]
                    
                p = re.compile("(\S*\s+)?([A-Za-z0-9_\.]+)\s+([A-Za-z0-9_\.]+)\s*=")
                side_matches = p.findall(var_and_type)
                if len(side_matches) == 0:
                    continue
                else:
                    for side_match in side_matches:
                        _, _type, var = side_match
                        add_dict(var_type_dict, var, _type, stmt_i)

        if stmt_i == 3: # fn_call_stmt
            for match in matches:
                constructor_new, var_dot_method = match
                if constructor_new == '': 
                    splitted = var_dot_method.split(".")
                    if len(splitted) >= 3:
                        method = splitted[-1]
                        var_or_module = ".".join(splitted[:-1])
                    elif len(splitted) == 2:
                        method = splitted[-1]
                        var_or_module = splitted[0]
                    elif len(splitted) == 1:
                        method = splitted[-1]
                        var_or_module = ""
                    else:
                        continue
                    add_dict(fn_var_dict, clean_method_name(method), var_or_module, stmt_i)
        
        if stmt_i == 4: # fn_call_stmt_1
            for match in matches:
                try:
                    constructor_new, var_dot_method = match
                    method = var_dot_method.split(".")[-1]
                    pos = get_method_pos(constructor_new)
                        
                    p = re.compile("new\s+([^(]+)[(]")
                    side_matches = p.findall(constructor_new)
                    var_or_module = side_matches[pos]
                    add_dict(fn_var_dict, clean_method_name(method), var_or_module, stmt_i)
                except Exception as e:
                    print(match)
                    raise e
    
    return var_type_dict, fn_var_dict, import_dict

import json
class_method_dict_dir = "/app/facos/data/class_method_dict.json"
with open(class_method_dict_dir, "r") as class_methods_dict_fp:
    class_methods_dict = json.load(class_methods_dict_fp)


def resolve_imports(import_dict):
    def append_value_to_key(key, value, dict):
        if key not in dict:
            dict[key] = []
        if value not in dict[key]:
            dict[key].append(value)

    dep_tracing_dict = {}
    
    for key in import_dict.keys():
        last_part = key.split(".")[-1]
        pre_part = ".".join(key.split(".")[:-1])
        
        term_first_char = [term[0] for term in key.split(".")]
        class_term_idx = None
        for term_i, char in enumerate(term_first_char):
            if char == char.upper():
                class_term_idx = term_i
                break
        if class_term_idx is None:
            
            if last_part == "*":
                append_value_to_key(key=last_part, value=pre_part, dict=dep_tracing_dict)
            else:
                raise Exception(f"error while adding wildcard_imports with `{pre_part}`")
        else:
            # _type = ".".join(key.split(".")[:class_term_idx+1])
            _type = key.split(".")[class_term_idx]
            _type_fqn = ".".join(key.split(".")[:class_term_idx+1])
            if not _type == last_part: # method or wildcard method is imported
                try:
                    assert _type_fqn+"."+last_part == key
                except Exception as e:
                    # print("Type assertion error:")
                    # print("key:", key)
                    # print("_type", _type)
                    # print("last_part", last_part)
                    return None
                
                if last_part == "*":
                    if _type_fqn in class_methods_dict:
                        for method in class_methods_dict[_type_fqn]:
                            append_value_to_key(key=method, value=_type_fqn, dict=dep_tracing_dict)
                            
                else: # method is imported, last_part is a method
                    append_value_to_key(key=last_part, value=_type_fqn, dict=dep_tracing_dict)

                    pass
            else: # Class is imported
                append_value_to_key(key=last_part, value=key, dict=dep_tracing_dict)
        
    return dep_tracing_dict        
        
def add_method(method_simple_name, _type, single_type_method={}, multi_type_method={}):
    if method_simple_name in multi_type_method:
        multi_type_method[method_simple_name].append(_type)
        return 
    if method_simple_name in single_type_method:
        previous_type = single_type_method[method_simple_name]
        if previous_type != _type:
            single_type_method.pop(method_simple_name)
            multi_type_method[method_simple_name] = [previous_type, _type]
    else:
        single_type_method[method_simple_name] = _type
    return single_type_method, multi_type_method


def trace_name(var_type_dict, fn_var_dict):
    single_type_method = {}
    multi_type_method = {}
    for method_simple_name, var_dict in fn_var_dict.items():
        for var in var_dict.keys():
            if var in var_type_dict:
                type_candidates = list(var_type_dict[var].keys())
                for candidate in type_candidates:
                    single_type_method, multi_type_method = add_method(method_simple_name, candidate, single_type_method, multi_type_method)
    
    return single_type_method, multi_type_method

def determine_var_package(var_type_dict, fn_var_dict):
    single_type_method = {}
    multi_type_method = {}
    for method_simple_name, var_dict in fn_var_dict.items():
        for var, var_stmt_i in var_dict.items():
            if var_stmt_i == 3: # extracted from stmt 3
                if var in var_type_dict: # if var is a variable
                    type_candidates = list(var_type_dict[var].keys())
                    for candidate in type_candidates:
                        add_method(method_simple_name, candidate, single_type_method, multi_type_method)
                else: # if var is a package/domain
                    if ("(" not in var) and (len(var.split(".")) > 1):
                        add_method(method_simple_name, var, single_type_method, multi_type_method)
                    elif (len(var.split(".")) == 1) and (var[0].isupper()):
                        add_method(method_simple_name, var, single_type_method, multi_type_method)
                          
            elif var_stmt_i in [4, 5]:
                add_method(method_simple_name, var, single_type_method, multi_type_method)
    
    return single_type_method, multi_type_method