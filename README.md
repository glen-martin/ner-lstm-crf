# Named entity recognition with BiLSTM 
Prototyping an NER system with Keras and Tensorflow employing a BiLSTM
Dataset from https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus

CRF implementation was removed as the keras_contrib layers are not properly integrated with tf.keras layers. 
Using tf_keras_contrib also does not solve this issue.
TFlite conversion possible by depending on tensorflow-nightly to fix issue with `Embedding` layer conversion 
Fully working on TF 2.0
WIP for converting model into TFLite
