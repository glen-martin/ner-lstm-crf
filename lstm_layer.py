import tensorflow as tf


def buildLstmLayer(inputs, num_layers, num_units):
    """Build the lstm layer.
  Args:
    inputs: The input data.
    num_layers: How many LSTM layers do we want.
    num_units: The unmber of hidden units in the LSTM cell.
  """
    lstm_cells = []
    for i in range(num_layers):
        lstm_cells.append(
            tf.lite.experimental.nn.TFLiteLSTMCell(
                num_units, forget_bias=0, name='rnn{}'.format(i)))
    lstm_layers = tf.keras.layers.StackedRNNCells(lstm_cells)
    # Assume the input is sized as [batch, time, input_size], then we're going
    # to transpose to be time-majored.
    transposed_inputs = tf.transpose(
        inputs, perm=[1, 0, 2])
    outputs, _ = tf.lite.experimental.nn.dynamic_rnn(
        lstm_layers,
        transposed_inputs,
        dtype='float32',
        time_major=True)
    unstacked_outputs = tf.unstack(outputs, axis=0)
    return unstacked_outputs[-1]
