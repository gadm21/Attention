

import tensorflow as tf

from tensorflow.keras.layers import Input, Embedding, Bidirectional, Dense, LSTM, Concatenate
from tensorflow import keras

max_len = 200
rnn_cell_size = 128
vocab_size=250

class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
    def call(self, features, hidden):
        print("features:{}".format(features.shape) ) 
        print("hidden:{}".format(hidden.shape))
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights



sequence_input = Input(shape=(max_len,), dtype='int32')

embedded_sequences = Embedding(vocab_size, 128, input_length=max_len)(sequence_input)

lstm = Bidirectional(LSTM (rnn_cell_size,
                                dropout=0.3,
                                return_sequences=True,
                                return_state=True,
                                recurrent_activation='relu',
                                recurrent_initializer='glorot_uniform'), name="bi_lstm_0")(embedded_sequences)

lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional( LSTM(rnn_cell_size,
      dropout=0.2,
      return_sequences=True,
      return_state=True,
      recurrent_activation='relu',
      recurrent_initializer='glorot_uniform'))(lstm)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])

#  PROBLEM IN THIS LINE
context_vector, attention_weights = Attention(units=32)(lstm, state_h)

output = Dense(1, activation='sigmoid')(context_vector)

model = keras.Model(inputs=sequence_input, outputs=output)

# summarize layers
print(model.summary())