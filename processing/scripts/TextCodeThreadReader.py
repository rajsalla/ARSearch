import os
import time
import re
import traceback
import csv
import json
import traceback


class TextCodeThreadReader:
    def __init__(self, content, thread_id, title_dict, tag_dict, ext=None):
        self.thread_id = thread_id
        self.file_ext = ext
        self.tags = ";".join(tag_dict[thread_id])
        self.title = title_dict[thread_id]
        content = "\n".join(content.split("\n")[1:])
        self.api_mentions = [] # self.api_mentions is built up after running remove_lables() of self.texts and self.code_blocks
        self.texts_w_labels, self.code_blocks_w_labels = self.get_text_and_code(content)
        self.texts = self._remove_labels(self.texts_w_labels, "text")
        self.code_blocks = self._remove_labels(self.code_blocks_w_labels, "code")

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
            
            return_par = par
            list_paragraphs = [return_par for _ in range(len(list_labeled_token_pos))]
            list_thread_ids = [self.thread_id for _ in range(len(list_labeled_token_pos))]
            content_types = [content_type for _ in range(len(list_labeled_token_pos))]
            api_mentions += zip(list_labeled_simple_names, list_labels)
            label_removed_content.append(return_par)

        self.api_mentions += api_mentions
        return label_removed_content
        

    def get_text(self):
        return "\n".join([line for line in self.texts if line.strip() != ""])
    
    def get_text_wo_label(self):
        return "\n".join([self.remove_api_tag(line) for line in self.texts if line.strip() != ""])
        
    def get_code(self):
        return self.list_code_block.get_code()
    
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
        texts = [text_line for text_line in texts if text_line.strip("\n") != ""]
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

    