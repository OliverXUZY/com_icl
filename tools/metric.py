class EditDistance:
    '''
    
    Customized implementation of Levenshtein Distance, (edit distance in leetcode)
    '''
    def minDistance(self, sen1: list, sen2: list) -> int:
        assert type(sen1) == list and type(sen2) == list, "split words in sentence into list"
        m = len(sen1); n = len(sen2)
        res = [[0] * (n + 1) for _ in range(m + 1)]
        res[0] = [j for j in range(n+1)]
        for i in range(m+1):
            res[i][0] = i
        for i in range(1, m+1):
            for j in range(1, n+1):
                substution = res[i-1][j-1] if sen1[i-1] == sen2[j-1] else res[i-1][j-1] + 1
                res[i][j] = min(res[i][j-1] + 1, res[i-1][j] + 1, substution)
        return res[m][n]
    
    def acc(self, ref: list, pre: list) -> int:
        dist = self.minDistance(ref, pre)
        wer = dist/len(ref)
        return wer



import rapidfuzz
class WER:
    '''

    WER implementation of Levenshtein Distance
    ### wer.py in evaluate huggingface --> jiwer --> RapidFuzz

    1. check wer in evaluate: https://huggingface.co/spaces/evaluate-metric/wer/blob/main/wer.py, it invokes jiwer
    2. check jiwer, it invokes rapidfuzz.distance.Levenshtein.editops


    This implementation is simplified from
    https://github.com/jitsi/jiwer/blob/7dfd00850aa9b0d46f14c2a0c82804cf0070270d/jiwer/process.py#L133

    '''
    def _word2char(self, reference, hypothesis):
        # for id, item in enumerate(hypothesis):
        #     if item == "":
        #         print(id)
        # print(hypothesis)
        # tokenize each word into an integer
        vocabulary = set(reference + hypothesis)

        if "" in vocabulary:
            raise ValueError(
                "Empty strings cannot be a word. "
                "Please ensure that the given transform removes empty strings."
            )

        word2char = dict(zip(vocabulary, range(len(vocabulary))))

        reference_chars = "".join([chr(word2char[w]) for w in reference])
        
        hypothesis_chars = "".join([chr(word2char[w]) for w in hypothesis])

        return reference_chars, hypothesis_chars

    def _process(self, reference, hypothesis):
        assert type(reference) == list and type(hypothesis) == list, "split words in sentence into list"

        ref, pred = self._word2char(reference, hypothesis)

        edit_ops = rapidfuzz.distance.Levenshtein.editops(ref, pred)
        substitutions = sum(1 if op.tag == "replace" else 0 for op in edit_ops)
        deletions = sum(1 if op.tag == "delete" else 0 for op in edit_ops)
        insertions = sum(1 if op.tag == "insert" else 0 for op in edit_ops)
        hits = len(ref) - (substitutions + deletions)

        # Compute all measures
        S, D, I, H = substitutions, deletions, insertions, hits

        ops = S + D + I
        wer = float(S + D + I) / float(H + S + D)

        return ops, wer
    
    def minDistance(self, reference, hypothesis):
        
        ops, wer = self._process(reference, hypothesis)
        
        return ops
    
    def acc(self, reference, hypothesis):
        ops, wer = self._process(reference, hypothesis)
        
        return wer


