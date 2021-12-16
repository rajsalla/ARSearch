def eval_mentions(text_mentions):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    count_case_in_cand = 0
    count_case_not_in_cand = 0
    tp_cases = []
    fp_cases = []
    fn_cases = []
    tn_cases = []
    for mention in text_mentions:
        pred = mention['pred']
        label = mention['label']
        
        if label == "None":
            count_case_not_in_cand += 1
        else:
            count_case_in_cand += 1
            
        if pred == 'None':
            if label == "None":
                tn += 1
                tn_cases.append(mention)
            else:
                fn += 1
                fn_cases.append(mention)
        else:
            if pred == label:
                tp += 1
                tp_cases.append(mention)
                
            else:
                fp += 1
                fp_cases.append(mention)
    
    if tp+fp == 0:
        prec = 0
    else:
        prec = tp/(tp+fp)
    if tp +fn == 0:
        recall = 0
    else:
        recall = tp/(tp+fn)
    print("precison: ", prec)
    print("recall: ", recall)
    if prec+recall == 0:
        f1 = -1
    else:
        f1 = 2*prec*recall/(prec+recall)
    print("F1-score: ", f1)