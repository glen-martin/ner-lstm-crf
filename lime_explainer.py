from eli5.lime import TextExplainer
from eli5.lime.samplers import MaskingTextSampler

from corpus_provider import Data
from ner_explainer import NERExplainerGenerator

data = Data()
labels = data.labels
sentences = data.sentences

index = 46781
label = labels[index]
text = sentences[index]
print(text)
print()
print(" ".join([f"{t} ({l})" for t, l in zip(text.split(), label)]))

for i, w in enumerate(text.split()):
    print(f"{i}: {w}")


def explain_predictions(model, word2idx, tag2idx, max_len):
    explainer_generator = NERExplainerGenerator(model, word2idx, tag2idx, max_len)
    word_index = 4
    predict_func = explainer_generator.get_predict_function(word_index=word_index)
    sampler = MaskingTextSampler(
        replacement="UNK",
        max_replace=0.7,
        token_pattern=None,
        bow=False
    )
    samples, similarity = sampler.sample_near(text, n_samples=4)
    print(samples)
    te = TextExplainer(
        sampler=sampler,
        position_dependent=True,
        random_state=42
    )
    te.fit(text, predict_func)
    te.explain_prediction(
        target_names=list(explainer_generator.idx2tag.values()),
        top_targets=3
    )
