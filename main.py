from corpus_provider import Data
from lime_explainer import explain_predictions
from ner_w_lime import build_bilstm_rmsprop_model, prepare_data, test_model
from utils import extract_tflite_save_model


if __name__ == '__main__':
    data = Data()
    idx2tag = data.idx2tag
    idx2word = data.idx2word
    word2idx = data.word2idx
    tag2idx = data.tag2idx
    max_len = data.max_len
    X_tr, X_te, y_tr, y_te = prepare_data(data)
    model = build_bilstm_rmsprop_model(data, X_tr, y_tr)
    test_model(model, X_te, y_te, idx2tag, idx2word)
    explain_predictions(model, word2idx, tag2idx, max_len)
    extract_tflite_save_model(model, model_name='ner_small')
