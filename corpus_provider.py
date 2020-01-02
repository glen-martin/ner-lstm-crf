from collections import Counter
import pandas as pd
from sentencegetter import SentenceGetter


class Data(object):
    def __init__(self):
        self.corpus_loc = "~/Projects/corpus/entity-annotated-corpus/ner_dataset.csv"

        self.data = pd.read_csv(self.corpus_loc, encoding="latin1").fillna(method="ffill")

        self.getter = SentenceGetter(self.data)
        sentences = self.getter.sentences

        self.labels = [[s[2] for s in sent] for sent in sentences]
        self.sentences = [" ".join([s[0] for s in sent]) for sent in sentences]

        word_cnt = Counter(self.data["Word"].values)
        vocabulary = set(w[0] for w in word_cnt.most_common(5000))

        words = list(set(self.data["Word"].values))
        tags = list(set(self.data["Tag"].values))

        self.max_len = 50

        self.word2idx = {"PAD": 0, "UNK": 1}
        self.word2idx.update({w: i for i, w in enumerate(words) if w in vocabulary})
        self.tag2idx = {t: i for i, t in enumerate(tags)}

        # Vocabulary Key:tag_index -> Value:Label/Tag
        self.idx2tag = {i: w for w, i in self.tag2idx.items()}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.n_words = len(words)
        self.n_tags = len(tags)

    def count_data_points(self):
        words = list(set(self.data["Word"].values))
        n_words = len(words)
        print(n_words)

        tags = list(set(self.data["Tag"].values))
        n_tags = len(tags)
        print(n_tags)
