import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, SpatialDropout1D, Bidirectional

from corpus_provider import Data


def build_bilstm_rmsprop_model(data: Data, X_tr, y_tr):
    max_len = data.max_len
    n_words = data.n_words
    n_tags = data.n_tags

    word_input = Input(shape=(max_len,))
    model = Embedding(input_dim=n_words, output_dim=50, input_length=max_len)(word_input)
    model = SpatialDropout1D(0.1)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)

    model = Model(word_input, out)
    model.compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(X_tr, y_tr.reshape(*y_tr.shape, 1),
                        batch_size=32, epochs=25,
                        validation_split=0.1, verbose=1)
    return model


def test_model(model, X_te, y_te, idx2tag, idx2word):
    pred_cat = model.predict(X_te)
    pred = np.argmax(pred_cat, axis=-1)
    y_te_true = np.argmax(y_te, -1)

    # Convert the index to tag
    pred_list = list()
    sentence_list = list()
    pred_tag = [[idx2tag[i] for i in row] for row in pred]
    y_te_true_tag = [[idx2tag[i] for i in row] for row in y_te]
    words = [[idx2word[i] for i in row] for row in X_te]

    total_sentences = list()
    for sentence, tag_list in zip(words, pred_tag):
        final_sent = ""
        for word, tag in zip(sentence, tag_list):
            if word != 'PAD':
                if tag != 'O':
                    final_sent += f" {word} ({tag})"
                else:
                    final_sent += " " + word

        total_sentences.append(final_sent)

    print('Padded sequences are ')
    print("\n".join(total_sentences))
    # print(y_te_true_tag)


def prepare_data(data: Data):
    word2idx = data.word2idx
    tag2idx = data.tag2idx
    sentences = data.sentences
    labels = data.labels
    max_len = data.max_len

    X = [[word2idx.get(w, word2idx["UNK"]) for w in s.split()] for s in sentences]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=word2idx["PAD"])

    y = [[tag2idx[l_i] for l_i in l] for l in labels]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, shuffle=False)
    return X_tr, X_te, y_tr, y_te
