class ApiMention:
    def __init__(self, api, label, sent_no, start_pos, end_pos):
        self.api = api
        self.label = label
        self.sent_no = sent_no
        self.start_pos = start_pos
        self.end_pos = end_pos
    
    def __repr__(self):
        return f"{(self.api, self.label, self.sent_no, self.start_pos, self.end_pos)}"
    
    def to_dict(self):
        return {
            'api': self.api,
            'label': self.label,
            'sent_no': self.sent_no,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos
        }

class ApiMentionInfer:
    def __init__(self, api, sent_no, start_pos, end_pos):
        self.api = api
        self.sent_no = sent_no
        self.start_pos = start_pos
        self.end_pos = end_pos
    
    def __repr__(self):
        return f"{(self.api, self.sent_no, self.start_pos, self.end_pos)}"
    
    def to_dict(self):
        return {
            'api': self.api,
            'sent_no': self.sent_no,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos
        }



