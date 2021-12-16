

class BaselineResult():
    def __init__(self, fqn, labels, ranked_threads, topk=5):
        self.fqn = fqn
        self.labels = labels
        # if len(self.labels) == 0:
        #     raise "err"
        self.ranked_threads = ranked_threads
        self.total_retrieved_threads = len(ranked_threads)
        # self.AP = self._calc_AP_at_cutoff(topk=topk)
        # self.MR = self._calc_MR_at_cutoff(topk=topk)
        # self.hit = self._calc_hit_at_cutoff(topk=topk)
        self.prec = self._calc_precision()
        self.recall = self._calc_recall()
        self.f1 = self._calc_f1()
#         print("AP: ", self.AP)
#         print("MR: ", self.MR)
#         print("hit: ", self.hit)
        
    def _calc_precision(self):
        predicted_positive = 0
        if len(self.ranked_threads) == 0: # tp == fp == 0
            if len(self.labels) == 0: #label == tp + fn => fn == 0
                return 1
            else:
                return 0
        for thread_id in self.ranked_threads:
            if thread_id in self.labels:
                predicted_positive += 1
        return predicted_positive/len(self.ranked_threads)
    
    def _calc_recall(self):
        """ Recall = tp/(tp+fn)
            Edge case when TP+FN == 0

        """
        true_positives = 0
        false_positives = 0
        
        for thread_id in self.ranked_threads:
            if thread_id in self.labels:
                true_positives += 1
            else:
                false_positives += 1
        if len(self.labels) == 0: #tp + fn == 0
            if false_positives == 0:
                return 1
            else:
                return 0
        return true_positives/len(self.labels)
    
    def _calc_f1(self):
        if self.prec == 0 and self.recall == 0:
            return 0
        return 2*self.prec *self.recall/(self.prec+self.recall)
        
    def _calc_hit_at_cutoff(self, topk=5):
        if len(self.ranked_threads) == 0:
            return 0
        
        if topk <= len(self.ranked_threads):
            topk_threads = self.ranked_threads[:topk]
        else:
            topk_threads = self.ranked_threads
        for thread_id in topk_threads:
            if thread_id in self.labels:
                return 1
        return 0
    
    def _calc_prec_at_cutoff(self, topk=5):
        if len(self.ranked_threads) == 0:
            return 0
        
        if topk <= len(self.ranked_threads):
            topk_threads = self.ranked_threads[:topk]
        else:
            topk_threads = self.ranked_threads
            
        predicted_positive = 0
        for thread_id in topk_threads:
            if thread_id in self.labels:
                predicted_positive += 1
        return predicted_positive/len(topk_threads)
    
    def _is_rel_at_k(self, k):
        if k  >= len(self.ranked_threads):
            return 0
        if self.ranked_threads[k] in self.labels:
            return 1
        else:
            return 0
    def _calc_AP_at_cutoff(self, topk=5):
        sum_prec = 0
        for i in range(topk):
            sum_prec += self._calc_prec_at_cutoff(i+1)*self._is_rel_at_k(i)

        return sum_prec/min(len(self.labels), topk)
            
        
    def _calc_recall_at_cutoff(self, topk=5):
        if len(self.ranked_threads) == 0:
            return 0
        
        if topk <= len(self.ranked_threads):
            topk_threads = self.ranked_threads[:topk]
        else:
            topk_threads = self.ranked_threads
            
        predicted_positive = 0
        for thread_id in topk_threads:
            if thread_id in self.labels:
                predicted_positive += 1
        return predicted_positive/len(self.labels)
    
    def _calc_MR_at_cutoff(self, topk=5):
        sum_prec = 0
        
        for i in range(min(topk, len(self.ranked_threads))):
            sum_prec += self._calc_recall_at_cutoff(i+1)*self._is_rel_at_k(i)

        return sum_prec/min(len(self.labels), topk)

    def get_fp_cases(self):
        fps = []
        for thread in self.ranked_threads:
            if thread not in self.labels:
                fps.append(thread)
        return fps

    def get_fn_cases(self):
        fns = []
        for thread in self.labels:
            if thread not in self.ranked_threads:
                fns.append(thread)
        return fns

    def get_fn_details(self):
        pass

    def get_tp_cases(self):
        tps = []
        for thread in self.ranked_threads:
            if thread in self.labels:
                tps.append(thread)

        return tps
    