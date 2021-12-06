import os
import time
import re
import traceback
import csv
import json
import traceback


class TextCodeThreadReader:
    def __init__(self, content, thread_id, title_dict, tag_dict, ext=None):
        # thread_id = content.split("\n")[0].split("/")[-1]
        # self.concern_apis_dict = self._get_concern_apis()
        self.thread_id = thread_id
        self.file_ext = ext
        self.tags = ";".join(tag_dict[thread_id])
        self.title = title_dict[thread_id]
        content = "\n".join(content.split("\n")[1:])
        self.api_mentions = [] # self.api_mentions is built up after running remove_lables() of self.texts and self.code_blocks
        self.texts_w_labels, self.code_blocks_w_labels = self.get_text_and_code(content)
        self.texts = self._remove_labels(self.texts_w_labels, "text")
        self.code_blocks = self._remove_labels(self.code_blocks_w_labels, "code")
        # self.mentioned_simple_names, self.mentioned_fqns = self._extract_mentioned_methods(self.api_mentions)
        
    # def _get_concern_apis(self):
    #     """
    #         Return a dictionary contains simple method names as keys and fqns as values
    #         {"simple_name": ["api1.simple_name", "api2.simple_name"], ...}
    #     """
    #     concern_api_dict = {}
    #     with open("apis.txt", "r") as fp:
    #         apis = fp.readlines()
    #     for api in apis:
    #         api = api.strip()
    #         simple_name = api.split(".")[-1]
    #         if simple_name not in concern_api_dict:
    #             concern_api_dict[simple_name] = []
    #         if api not in concern_api_dict[simple_name]:
    #             concern_api_dict[simple_name].append(api)
    #     return concern_api_dict

    def get_text_and_code(self, content):
        texts = self.find_text_with_regex(content)
        code_blocks = self.find_code_block_with_regex(content)
        return texts, code_blocks

    def remove_api_tag(self, text):
        pattern = '<API label="(.*?)">(.*?)</API>'
        text = re.sub(pattern,  "\g<2>", text)
        return text
    
    def _tokenize(self, content):
        tokens = re.findall("\w+(?:'\w+)?|[^\w\s]",content)
        return tokens
    def _remove_labels(self, content, content_type):
        label_removed_content = []
        pattern = '<API label="(.*?)">(.*?)</API>'
        api_mentions = []
        for par in content:
            par = par.replace("<code>", "")
            par = par.replace("</code>", "")
            par = par.replace("<pre>", "")
            par = par.replace("</pre>", "")
            par = par.replace("==========", "")
            match = re.search(pattern, par)
            list_labeled_simple_names = []
            list_labeled_token_pos = []
            list_paragraphs = []
            list_labels = []
            while match is not None:
                s = match.start()
                e = match.end()
                content_before_api = par[:s]
                before_api_tokens = self._tokenize(content_before_api)
                matching_tag = match.group(0)
                label =  match.group(1)
                api = match.group(2)
                # print(matching_tag)
                api_mention_tokens = self._tokenize(api)
                if len(api_mention_tokens) > 0:
                    simple_name = api_mention_tokens[-1]
                    list_labeled_simple_names.append(simple_name)
                    list_labels.append(label)

                par = re.sub(re.escape(matching_tag), api, par, 1)
                match = re.search(pattern, par)
            
            # return_par = " ".join(self._tokenize(par))
            return_par = par
            list_paragraphs = [return_par for _ in range(len(list_labeled_token_pos))]
            list_thread_ids = [self.thread_id for _ in range(len(list_labeled_token_pos))]
            content_types = [content_type for _ in range(len(list_labeled_token_pos))]
            api_mentions += zip(list_labeled_simple_names, list_labels)
            label_removed_content.append(return_par)

        self.api_mentions += api_mentions
        return label_removed_content
        
    # def _extract_mentioned_methods(self, api_mentions):
    #     simple_names = []
    #     fqns = []
    #     for simple_name, fqn in api_mentions:
    #         # print(simple_name)
    #         if simple_name[0] == simple_name[0].upper():
    #             continue
    #         if simple_name not in simple_names:
    #             simple_names.append(simple_name)
    #         if fqn != "None" and fqn != "":
    #             try:
    #                 if simple_name not in self.concern_apis_dict:
    #                     continue
    #                 if fqn in self.concern_apis_dict[simple_name]:
    #                     if fqn not in fqns:
    #                         fqns.append(fqn)
    #             except Exception as e:
    #                 traceback.print_exc()
    #                 print(self.thread_id)
    #                 print(fqn)
    #                 print(simple_name)
    #                 print(list(self.concern_apis_dict.keys()))
    #                 exit()
    #     return simple_names, fqns

    # def get_text_code_pairs(self):
    #     # for sname in self.mentioned_simple_names:
    #     pairs = []
    #     for text in self.texts:
    #         for code in self.code_blocks:
    #             pair = [text, code]
    #             pairs.append(pair)
    #     text_code_pairs = []
    #     for pair in pairs:
    #         # all_tokens = self._tokenize(pair[0]) + self._tokenize(pair[1])
    #         for sname in self.mentioned_simple_names:
    #             if sname not in self.concern_apis_dict:
    #                 continue
                
    #             # if sname == "print" and self.thread_id == "51558046":
    #             #     continue

    #             # positive_target means FQNs mentioned in the thread
    #             positive_targets = [fqn for fqn in self.mentioned_fqns if fqn.split(".")[-1]==sname]
    #             # if self.thread_id=="50550694" and sname=="verify":
    #             #     print(positive_targets)
    #             #     exit()
    #             for positive_target in [fqn for fqn in self.mentioned_fqns if fqn.split(".")[-1]==sname]:
    #                 text_code_pairs.append((sname, positive_target, True, self.thread_id, pair))

    #             # negative_target means FQNs not mentioned in thread but their simple name are
    #             # candidate_fqns is a list of all given FQN APIs having the simple name `sname`
    #             try:
    #                 candidate_fqns = self.concern_apis_dict[sname]
    #             except Exception as e:
    #                 traceback.print_exc()
    #                 print(self.thread_id)
    #                 print(sname)
    #                 exit()
    #             negative_fqns = [_fqn for _fqn in candidate_fqns if _fqn not in self.mentioned_fqns]
    #             # if self.thread_id == "50550694" and sname=="verify":
    #             #     print(negative_fqns)
    #             #     exit()
    #             for negative_target in negative_fqns:
    #                 text_code_pairs.append((sname, negative_target, False, self.thread_id, pair))
    #     return text_code_pairs


    def get_text(self):
        return "\n".join([line for line in self.texts if line.strip() != ""])
    
    def get_text_wo_label(self):
        return "\n".join([self.remove_api_tag(line) for line in self.texts if line.strip() != ""])
        
    def get_code(self):
        return self.list_code_block.get_code()
        # return "\n".join([blk.get() for blk in self.list_code_block.list])
    
    def get_code_with_label(self):
        return self.list_code_block.get_code_with_label()
    
    def get_tags(self):
        return self.tags
    
    def get_title(self):
        return self.title

    def find_text_with_regex(self, content):
        replace_downline_char = " _downline_datys_ "
        content = content.replace("\n", replace_downline_char)
        if self.file_ext == "txt":
            regex = "<code>.*?</code>"
        else:
            regex = "<pre><code>.*?</code></pre>"
        texts = re.split(regex, content)
        for i in range(len(texts)):
            texts[i] = texts[i].replace(replace_downline_char, "\n")
        # print(texts)
        texts = [text_line for text_line in texts if text_line.strip("\n") != ""]
        # print(texts)
        # print(texts)
        return texts

    def find_code_block_with_regex(self, content):
        code_blocks = []
        replace_downline_char = " _downline_datys_ "
        content = content.replace("\n", replace_downline_char)
        
        if self.file_ext == "txt":
            regex = "<code>.*?</code>"
        else:
            regex = "<pre><code>.*?</code></pre>"
        regex_blocks = re.findall(regex, content)
        for block in regex_blocks:
            if self.file_ext == "txt":
                block = block[:6]  + "\n" +block[6:-7] + "\n" + block[-7:]
            else:
                block = block[:11]  + "\n" +block[11:-13] + "\n" + block[-13:]
            block = block.replace(replace_downline_char, "\n")
            code_blocks.append(block)
        return code_blocks

    