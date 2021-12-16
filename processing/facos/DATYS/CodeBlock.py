import re

class CodeBlock:
    def __init__(self, code, start_at_line, end_at_line):
        self.code_with_label = code
        self.code = self.remove_api_tag(code)
        self.start = start_at_line
        self.end = end_at_line
        self.neighbor_text = None
    
    def remove_api_tag(self, code):
        pattern = '<API label="(.*?)">(.*?)</API>'
        code = re.sub(pattern,  "\g<2>", code)
        return code
    
    def get_line_number_in_block(self, line_number_in_content):
        if line_number_in_content > self.start and line_number_in_content < self.end:
            return line_number_in_content - self.start - 1
        else:
            return -1
    
    def get_code_line_in_block(self, api_mention):
        line_i = self.get_line_number_in_block(api_mention.sent_no)
        if line_i == -1:
            return ""
        else:
            return self.get_code_wo_tag().split("\n")[line_i]
        
    def get_code_wo_tag(self):
        return "\n".join(self.code.split("\n")[1:-1])

    def get_labeled_code_wo_tag(self):
        return "\n".join(self.code_with_label.split("\n")[1:-1])
    
    def add_neighbor_text(self, neighbor_text):
        self.neighbor_text = neighbor_text
    
    def __repr__(self):
        return "\n".join([f"{line_i:03d}: {line}" for line_i, line in enumerate(self.get_code_wo_tag().split("\n"))])
    
    def get(self):
        return self.get_code_wo_tag()

    
class ListCodeBlock:
    def __init__(self, code_blocks):
        self.list = code_blocks
    
    def get_code_block_of_mention(self, api_mention):
        for blk_i, code_block in enumerate(self.list):
            line_in_code = code_block.get_line_number_in_block(api_mention.sent_no)
            if line_in_code != -1:
                return True, blk_i, line_in_code
        return False, -1, -1