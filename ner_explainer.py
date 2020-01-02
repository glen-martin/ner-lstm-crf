from tensorflow.keras.preprocessing.sequence import pad_sequences

class NERExplainerGenerator(object):

    def __init__(self, model, word2idx, tag2idx, max_len):
        self.model = model
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.idx2tag = {v: k for k, v in tag2idx.items()}
        self.max_len = max_len

    def _preprocess(self, texts):
        X = [[self.word2idx.get(w, self.word2idx["UNK"]) for w in t.split()]
             for t in texts]
        X = pad_sequences(maxlen=self.max_len, sequences=X,
                          padding="post", value=self.word2idx["PAD"])
        return X

    def get_predict_function(self, word_index):
        def predict_func(texts):
            X = self._preprocess(texts)
            p = self.model.predict(X)
            return p[:, word_index, :]

        return predict_func