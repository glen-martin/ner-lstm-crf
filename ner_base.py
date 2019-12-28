import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF

from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from sentencegetter import SentenceGetter

model_name = 'ner-lstm-crf'


def extract_tflite_model():
    tensorflow.reset_default_graph()
    converter = tensorflow.lite.TFLiteConverter.from_keras_model_file(model_name + '.h5',
                                                                      custom_objects=create_custom_objects())
    tflite_model = converter.convert()
    open(model_name + '.tflite', 'wb').write(tflite_model)
    print('Model converted successfully!')


def create_custom_objects():
    instanceHolder = {"instance": None}

    class ClassWrapper(CRF):
        def __init__(self, *args, **kwargs):
            instanceHolder["instance"] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)

    def loss(*args):
        method = getattr(instanceHolder["instance"], "loss_function")
        return method(*args)

    def accuracy(*args):
        method = getattr(instanceHolder["instance"], "accuracy")
        return method(*args)

    return {"ClassWrapper": ClassWrapper, "CRF": ClassWrapper, "loss": loss, "accuracy": accuracy}


def load_keras_model(path):
    model = load_model(path, custom_objects=create_custom_objects())
    return model


def build_model():
    # Model definition
    input = Input(shape=(MAX_LEN,))
    model = Embedding(input_dim=n_words + 2, output_dim=EMBEDDING,  # n_words + 2 (PAD & UNK)
                      input_length=MAX_LEN)(input)  # default: 20-dim embedding
    model = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.1))(model)  # variational biLSTM
    model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    crf = CRF(n_tags + 1)  # CRF layer, n_tags+1(PAD)
    out = crf(model)  # output

    model = Model(input, out)
    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

    model.summary()
    return model


def train_model(model, X_tr, X_te, y_tr, y_te):
    history = model.fit(X_tr, np.array(y_tr), batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=2)
    # Eval
    pred_cat = model.predict(X_te)
    pred = np.argmax(pred_cat, axis=-1)
    y_te_true = np.argmax(y_te, -1)

    # Convert the index to tag
    pred_tag = [[idx2tag[i] for i in row] for row in pred]
    y_te_true_tag = [[idx2tag[i] for i in row] for row in y_te_true]

    report = flat_classification_report(y_pred=pred_tag, y_true=y_te_true_tag)
    print(report)


BATCH_SIZE = 512  # Number of examples used in each iteration
EPOCHS = 1  # Number of passes through entire dataset
MAX_LEN = 80  # Max length of review (in words)
EMBEDDING = 40  # Dimension of word embedding vector

corpus = "~/Projects/corpus/entity-annotated-corpus/ner_dataset.csv"

data = pd.read_csv(corpus, encoding="latin1")
data = data.fillna(method="ffill")

print("Number of sentences: ", len(data.groupby(['Sentence #'])))

words = list(set(data["Word"].values))
n_words = len(words)
print("Number of words in the dataset: ", n_words)

tags = list(set(data["Tag"].values))
print("Tags:", tags)
n_tags = len(tags)
print("Number of Labels: ", n_tags)

print("What the dataset looks like:" + str(data.head()))

getter = SentenceGetter(data)
sentences = getter.sentences

# Plot sentence by lenght
# plt.hist([len(s) for s in sentences], bins=50)
# plt.title('Token per sentence')
# plt.xlabel('Len (number of token)')
# plt.ylabel('# samples')
# plt.show()

# Vocabulary Key:word -> Value:token_index
# The first 2 entries are reserved for PAD and UNK
word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["UNK"] = 1  # Unknown words
word2idx["PAD"] = 0  # Padding

# Vocabulary Key:token_index -> Value:word
idx2word = {i: w for w, i in word2idx.items()}

# Vocabulary Key:Label/Tag -> Value:tag_index
# The first entry is reserved for PAD
tag2idx = {t: i + 1 for i, t in enumerate(tags)}
tag2idx["PAD"] = 0

# Vocabulary Key:tag_index -> Value:Label/Tag
idx2tag = {i: w for w, i in tag2idx.items()}

print("The word Obama is identified by the index: {}".format(word2idx["Obama"]))
print("The labels B-geo(which defines Geopraphical Enitities) is identified by the index: {}".format(tag2idx["B-geo"]))

# Convert each sentence from list of Token to list of word_index
X = [[word2idx[w[0]] for w in s] for s in sentences]
# Padding each sentence to have the same lenght
X = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post", value=word2idx["PAD"])

# Convert Tag/Label to tag_index
y = [[tag2idx[w[2]] for w in s] for s in sentences]
# Padding each sentence to have the same lenght
y = pad_sequences(maxlen=MAX_LEN, sequences=y, padding="post", value=tag2idx["PAD"])

# One-Hot encode
y = [to_categorical(i, num_classes=n_tags + 1) for i in y]  # n_tags+1(PAD)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
print(X_tr.shape, X_te.shape, np.array(y_tr).shape, np.array(y_te).shape)

print('Raw Sample: ', ' '.join([w[0] for w in sentences[0]]))
print('Raw Label: ', ' '.join([w[2] for w in sentences[0]]))

model = build_model()
train_model(model, X_tr, X_te, y_tr, y_te)
model.save(model_name + '.h5')
extract_tflite_model()
