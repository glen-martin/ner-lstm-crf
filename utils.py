import tensorflow


def extract_tflite_save_model(model, model_name):
    model.save(model_name+'.h5')
    converter = tensorflow.lite.TFLiteConverter.from_keras_model(model)
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    open(model_name + '.tflite', 'wb').write(tflite_model)
    print('Model converted successfully!')