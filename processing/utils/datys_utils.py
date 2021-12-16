from sklearn.feature_extraction.text import CountVectorizer

def tokenize(text):
    if text == "":
        return [], {}
    count_vec = CountVectorizer(lowercase=False)
    content_vocabs = count_vec.fit([text]).vocabulary_
    tokens= (list(content_vocabs.keys()))
    return tokens, content_vocabs