from re import L
import traceback
from bs4 import BeautifulSoup

from .utils.code_extractor import *
from .ApiMention import ApiMention, ApiMentionInfer
from .CodeBlock import CodeBlock, ListCodeBlock

class Thread:
    def __init__(self, thread_id, content, title, tags):
        # thread_id = content.split("\n")[0].split("/")[-1]
        
        content= content.split("\n")[1:]
        links = []

        new_content = []
        for sentence in content:
            if sentence[:11] == "<pre><code>":
                new_content.append(sentence[:11])
                new_content.append(sentence[11:])
            else:
                new_content.append(sentence)

        content = "\n".join(new_content)
        code_blocks = self.find_code_block(content)
        list_code_block = ListCodeBlock(code_blocks)
        texts = self.find_text(content)
        pattern = '<API label="(.*?)">(.*?)</API>'
        list_mention = []

        for sent_no, sentence in enumerate(content.split("\n")):
            match = re.search(pattern, sentence)
            while match is not None:
                s = match.start()
                matching_tag = match.group(0)
                label =  match.group(1)
                api = match.group(2)
                sentence = re.sub(re.escape(matching_tag), api, sentence, 1)
                list_mention.append(ApiMention(api, label, sent_no, s, s+len(api)))
                match = re.search(pattern, sentence)

        mention_codeblk_mapping = {}
        for mention_indice, mention in enumerate(list_mention):
            res, code_blk_indice, code_line = list_code_block.get_code_block_of_mention(mention)
            if res:
                mention_codeblk_mapping[mention_indice] = code_blk_indice

        for mention_i, code_blk_i in mention_codeblk_mapping.items():
            code_block = list_code_block.list[code_blk_i]
            api_mention = list_mention[mention_i]

        self.thread_id = thread_id
        self.texts = texts
        self.list_code_block = list_code_block
        self.list_mention = list_mention
        self.mention_codeblk_mapping = mention_codeblk_mapping
        self.tags = tags
        self.title = title
        
    def remove_api_tag(self, text):
        pattern = '<API label="(.*?)">(.*?)</API>'
        text = re.sub(pattern,  "\g<2>", text)
        return text
    
    def get_text(self):
        return "\n".join([line for line in self.texts if line.strip() != ""])
    
    def get_text_wo_label(self):
        return "\n".join([self.remove_api_tag(line) for line in self.texts if line.strip() != ""])
        
    def get_code(self):
        return "\n".join([blk.get() for blk in self.list_code_block.list])
    
    def get_tags(self):
        return self.tags
    
    def get_title(self):
        return self.title
    
    def find_code_block(self, content):
        blocks = []
        start = -1
        end = -1
        for line_idx, line in enumerate(content.split("\n")):
            if line.strip() == "<pre><code>":
                start = line_idx
            elif line.strip() == "</code></pre>":
                end = line_idx
            if (start != -1) and (end != -1):
                blocks.append(CodeBlock("\n".join(content.split("\n")[start:end+1]), start, end))
                start = -1
                end = -1
        return blocks

    def find_text(self, content):
        texts = []
        start = 0
        end = -1
        for line_idx, line in enumerate(content.split("\n")):
            if line.strip() == "<pre><code>":
                end = line_idx - 1
            elif line.strip() == "</code></pre>":
                start = line_idx + 1
            if (start != -1) and (end != -1):
                if start == 0 and end == 0:
                    pass
                else:
                    texts.append("\n".join([line for line in content.split("\n")[start:end+1] if (line.strip() != "" and line.strip() != "==========")]))
                start = -1
                end = -1
        text_end = []
        content_reverse = content.split("\n")
        content_reverse.reverse()
        return_content = None
        for line_idx, line in enumerate(content_reverse):
            if line.strip() == "</code></pre>":
                if line_idx == 0:
                    return texts
                else:
                    return_content = content_reverse[:line_idx-1]
                    return_content.reverse()
                    break
        if return_content is not None:
            return_content = [line for line in return_content if (line.strip() != "" and line.strip() != "==========")]
            return texts+return_content
        else:
            return texts

    def get_possible_type_dict(self):
        possible_types = {}
        var_type_dict, fn_var_dict, import_dict = extract_code(self.get_code())
        for key in import_dict.keys():
            parts = key.split(".")
            if parts[-1] == "*":
                continue
            possible_types[parts[-1]] = key
        
        for var, value in var_type_dict.items():
            _type = list(value.keys())[0]
            possible_types[var] = _type
        return possible_types
            
    def extract_possible_types(self):
        def update_dicts(dict_old, dict_new):
            from copy import deepcopy
            updated_dict = deepcopy(dict_old)
            for key, list_value in dict_new.items():
                if key not in updated_dict:
                    updated_dict[key] = []
                for item in list_value:
                    if item not in updated_dict[key]:
                        updated_dict[key].append(item)
            return updated_dict

        try:
            var_type_dict, fn_var_dict, import_dict = extract_code(self.get_code())
            dep_tracing_dict = resolve_imports(import_dict)
            fn_var_dict_trans = {}
            for fn, _vars in fn_var_dict.items():
                list_vars_calling_fn = [_var for _var in _vars.keys()]
                fn_var_dict_trans[fn] = list_vars_calling_fn
            single_type_method, multi_type_method = determine_var_package(var_type_dict, fn_var_dict)

            fn_type_dict = {}
            for key, value in single_type_method.items():
                if key not in multi_type_method:
                    multi_type_method[key] = []
                if value not in multi_type_method[key]:
                    multi_type_method[key].append(value)

            fn_type_dict = multi_type_method
            dependencies = update_dicts({}, fn_type_dict)
            dependencies = update_dicts(dependencies, dep_tracing_dict)
            return dependencies
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"Error with Thread {self.thread_id}")
            
    def get_api_mention_text(self):
        text = self.get_text()
        list_mentions_in_text = []
        for line_i, text_line in enumerate(text.split("\n")):
            soup = BeautifulSoup(text_line, 'html.parser')
            for api in soup.findAll("api"):
                if api['label'].strip() != "":
                    api_mention = {
                        'name': api.text,
                        'label': api['label'],
                        'line': soup.text,
                        'line_i': line_i,
                        'thread_id': self.thread_id
                    }
                    list_mentions_in_text.append(api_mention)
        return list_mentions_in_text

    def get_api_mention_infer(self):
        pass


class ThreadInfer:
    def __init__(self, thread_id, content, title, tags):
        # thread_id = content.split("\n")[0].split("/")[-1]
        
        content= content.split("\n")[1:]
        links = []

        new_content = []
        for sentence in content:
            if sentence[:11] == "<pre><code>":
                new_content.append(sentence[:11])
                new_content.append(sentence[11:])
            else:
                new_content.append(sentence)

        content = "\n".join(new_content)
        code_blocks = self.find_code_block(content)
        list_code_block = ListCodeBlock(code_blocks)
        texts = self.find_text(content)
        pattern = '<API label="(.*?)">(.*?)</API>'
        list_mention = []

        for sent_no, sentence in enumerate(content.split("\n")):
            match = re.search(pattern, sentence)
            while match is not None:
                s = match.start()
                matching_tag = match.group(0)
                # label =  match.group(1)
                api = match.group(2)
                sentence = re.sub(re.escape(matching_tag), api, sentence, 1)
                list_mention.append(ApiMentionInfer(api, sent_no, s, s+len(api)))
                match = re.search(pattern, sentence)

        mention_codeblk_mapping = {}
        for mention_indice, mention in enumerate(list_mention):
            res, code_blk_indice, code_line = list_code_block.get_code_block_of_mention(mention)
            if res:
                mention_codeblk_mapping[mention_indice] = code_blk_indice

        for mention_i, code_blk_i in mention_codeblk_mapping.items():
            code_block = list_code_block.list[code_blk_i]
            api_mention = list_mention[mention_i]

        self.thread_id = thread_id
        self.texts = texts
        self.list_code_block = list_code_block
        self.list_mention = list_mention
        self.mention_codeblk_mapping = mention_codeblk_mapping
        self.tags = tags
        self.title = title
        
    def remove_api_tag(self, text):
        pattern = '<API label="(.*?)">(.*?)</API>'
        text = re.sub(pattern,  "\g<2>", text)
        return text
    
    def get_text(self):
        return "\n".join([line for line in self.texts if line.strip() != ""])
    
    def get_text_wo_label(self):
        return "\n".join([self.remove_api_tag(line) for line in self.texts if line.strip() != ""])
        
    def get_code(self):
        return "\n".join([blk.get() for blk in self.list_code_block.list])
    
    def get_labeled_code(self):
        return "\n".join([blk.get_labeled_code_wo_tag() for blk in self.list_code_block.list])

    def get_tags(self):
        return self.tags
    
    def get_title(self):
        return self.title
    
    def find_code_block(self, content):
        blocks = []
        start = -1
        end = -1
        for line_idx, line in enumerate(content.split("\n")):
            if line.strip() == "<pre><code>":
                start = line_idx
            elif line.strip() == "</code></pre>":
                end = line_idx
            if (start != -1) and (end != -1):
                blocks.append(CodeBlock("\n".join(content.split("\n")[start:end+1]), start, end))
                start = -1
                end = -1
        return blocks

    def find_text(self, content):
        texts = []
        start = 0
        end = -1
        for line_idx, line in enumerate(content.split("\n")):
            if line.strip() == "<pre><code>":
                end = line_idx - 1
            elif line.strip() == "</code></pre>":
                start = line_idx + 1
            if (start != -1) and (end != -1):
                if start == 0 and end == 0:
                    pass
                else:
                    texts.append("\n".join([line for line in content.split("\n")[start:end+1] if (line.strip() != "" and line.strip() != "==========")]))
                start = -1
                end = -1
        text_end = []
        content_reverse = content.split("\n")
        content_reverse.reverse()
        return_content = None
        for line_idx, line in enumerate(content_reverse):
            if line.strip() == "</code></pre>":
                if line_idx == 0:
                    return texts
                else:
                    return_content = content_reverse[:line_idx-1]
                    return_content.reverse()
                    break
        if return_content is not None:
            return_content = [line for line in return_content if (line.strip() != "" and line.strip() != "==========")]
            return texts+return_content
        else:
            return texts

    def get_possible_type_dict(self):
        possible_types = {}
        var_type_dict, fn_var_dict, import_dict = extract_code(self.get_code())
        for key in import_dict.keys():
            parts = key.split(".")
            if parts[-1] == "*":
                continue
            possible_types[parts[-1]] = key
        
        for var, value in var_type_dict.items():
            _type = list(value.keys())[0]
            possible_types[var] = _type
        return possible_types
            
    def extract_possible_types(self):
        def update_dicts(dict_old, dict_new):
            from copy import deepcopy
            updated_dict = deepcopy(dict_old)
            for key, list_value in dict_new.items():
                if key not in updated_dict:
                    updated_dict[key] = []
                for item in list_value:
                    if item not in updated_dict[key]:
                        updated_dict[key].append(item)
            return updated_dict

        try:
            var_type_dict, fn_var_dict, import_dict = extract_code(self.get_code())
            dep_tracing_dict = resolve_imports(import_dict)
            fn_var_dict_trans = {}
            for fn, _vars in fn_var_dict.items():
                list_vars_calling_fn = [_var for _var in _vars.keys()]
                fn_var_dict_trans[fn] = list_vars_calling_fn
            single_type_method, multi_type_method = determine_var_package(var_type_dict, fn_var_dict)

            fn_type_dict = {}
            for key, value in single_type_method.items():
                if key not in multi_type_method:
                    multi_type_method[key] = []
                if value not in multi_type_method[key]:
                    multi_type_method[key].append(value)

            fn_type_dict = multi_type_method
            dependencies = update_dicts({}, fn_type_dict)
            dependencies = update_dicts(dependencies, dep_tracing_dict)
            return dependencies
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"Error with Thread {self.thread_id}")
            
    def get_api_mention_text(self):
        text = self.get_text()
        list_mentions_in_text = []
        for line_i, text_line in enumerate(text.split("\n")):
            soup = BeautifulSoup(text_line, 'html.parser')
            for api in soup.findAll("api"):
                if api['label'].strip() != "":
                    api_mention = {
                        'name': api.text,
                        'label': api['label'],
                        'line': soup.text,
                        'line_i': line_i,
                        'thread_id': self.thread_id
                    }
                    list_mentions_in_text.append(api_mention)
        return list_mentions_in_text
    
    def get_api_mention_text_and_code(self):
        text = self.get_text()
        code = self.get_labeled_code()
        list_mentions_in_text_and_code = []
        for line_i, text_line in enumerate(text.split("\n")):
            soup = BeautifulSoup(text_line, 'html.parser')
            for api in soup.findAll("api"):
                api_mention = {
                    'name': api.text,
                    'thread_id': self.thread_id
                }
                list_mentions_in_text_and_code.append(api_mention)
        for code_line in code.split("\n"):
            soup = BeautifulSoup(code_line, 'html.parser')
            for api in soup.findAll("api"):
                api_mention = {
                    'name': api.text,
                    'thread_id': self.thread_id
                }
                list_mentions_in_text_and_code.append(api_mention)
        return list_mentions_in_text_and_code

    def get_api_mention_infer(self):
        pass
    