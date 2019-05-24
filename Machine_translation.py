"""
Author: Arvind Ramesh
Reg No:R00171371
Msc.Artificial Intelligence
"""
"""Machine translation model: Built with reference to keras LSTM example
https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
"""
with open('deu.txt', 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
train_sentences = []
target_sentences = []
train_char = set()
target_char = set()
samples = 150000
# samples = 1000
for line in lines[: min(samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    target_text = '\t' + target_text + '\n'
    train_sentences.append(input_text)
    target_sentences.append(target_text)
    for char in input_text:
        if char not in train_char:
            train_char.add(char)
    for char in target_text:
        if char not in target_char:
            target_char.add(char)

print(train_sentences[99000])
print(target_sentences[99000])

train_char = sorted(list(train_char))
target_char = sorted(list(target_char))
encoder_token_len = len(train_char)
decoder_token_len = len(target_char)
total_length_encoder = max([len(txt) for txt in train_sentences])
total_length_decoder = max([len(txt) for txt in target_sentences])

print('Number of samples:', len(train_sentences))
print('Number of unique input tokens:', encoder_token_len)
print('Number of unique output tokens:', decoder_token_len)
print('Max sequence length for inputs:', total_length_encoder)
print('Max sequence length for outputs:', total_length_decoder)


tokenized_encoder_intvalue = dict(
  [(char, i) for i, char in enumerate(train_char)])
tokenized_decoder_intvalue = dict(
  [(char, i) for i, char in enumerate(target_char)])

import numpy as np

En_input = np.zeros(
  (len(train_sentences), total_length_encoder, encoder_token_len),
  dtype='float32')
De_input = np.zeros(
  (len(train_sentences), total_length_decoder, decoder_token_len),
  dtype='float32')
De_output = np.zeros(
  (len(train_sentences), total_length_decoder, decoder_token_len),
  dtype='float32')

print(En_input.shape)
print(De_input.shape)

for i, (input_text, target_text) in enumerate(zip(train_sentences, target_sentences)):
    for t, char in enumerate(input_text):
        En_input[i, t, tokenized_encoder_intvalue[char]] = 1.
    for t, char in enumerate(target_text):
        # De_output is ahead of De_input by one timestep
        De_input[i, t, tokenized_decoder_intvalue[char]] = 1.
        if t > 0:
            # De_output will be ahead by one timestep
            # and will not include the start character.
            De_output[i, t - 1, tokenized_decoder_intvalue[char]] = 1.

import keras, tensorflow
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

batch_size = 64  # batch size for training
epochs = 2  # number of epochs to train for
latent_dim = 256  # latent dimensionality of the encoding space

input_encoder = Input(shape=(None, encoder_token_len))
encoder = LSTM(latent_dim, return_state=True)
output_encoder, state_h, state_c = encoder(input_encoder)
encoder_states = [state_h, state_c]

input_decoder = Input(shape=(None, decoder_token_len))
decoder_lstm_model = LSTM(latent_dim, return_sequences=True, return_state=True)
output_decoder, _, _ = decoder_lstm_model(input_decoder,
                                     initial_state=encoder_states)
decoder_dense_layer = Dense(decoder_token_len, activation='softmax')
output_decoder = decoder_dense_layer(output_decoder)

model = Model(inputs=[input_encoder, input_decoder],
              outputs=output_decoder)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.summary()

# model.fit([En_input, De_input], De_output,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_split=0.2)
# model.save('abcd.h5')  #change the file to complete.h5 to get the trained weights


model = Model([input_encoder, input_decoder], output_decoder)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.load_weights("abcd.h5")

encoder_model = Model(input_encoder, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

output_decoder, state_h, state_c = decoder_lstm_model(
  input_decoder, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
output_decoder = decoder_dense_layer(output_decoder)

decoder_model = Model(
  [input_decoder] + decoder_states_inputs,
  [output_decoder] + decoder_states)

reverse_input_char_index = dict(
  (i, char) for char, i in tokenized_encoder_intvalue.items())
reverse_target_char_index = dict(
  (i, char) for char, i in tokenized_decoder_intvalue.items())


def decode_sequence(input_seq):
    # encode the input sequence to get the internal state vectors.
    states_value = encoder_model.predict(input_seq)

    # generate empty target sequence of length 1 with only the start character
    target_seq = np.zeros((1, 1, decoder_token_len))
    target_seq[0, 0, tokenized_decoder_intvalue['\t']] = 1.

    # output sequence loop
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # sample a token and add the corresponding character to the
        # decoded sequence
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # check for the exit condition: either hitting max length
        # or predicting the 'stop' character
        if (sampled_char == '\n' or
                len(decoded_sentence) > total_length_decoder):
            stop_condition = True

        # update the target sequence (length 1).
        target_seq = np.zeros((1, 1, decoder_token_len))
        target_seq[0, 0, sampled_token_index] = 1.

        # update states
        states_value = [h, c]

    return decoded_sentence

for seq_index in range(10):
    input_seq = En_input[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', train_sentences[seq_index])
    print('Decoded sentence:', decoded_sentence)

""" enter the sentence that needs to be translated"""
input_sentence = "i do not like this movie at all?"
test_sentence_tokenized = np.zeros(
  (1, total_length_encoder, encoder_token_len), dtype='float32')
for t, char in enumerate(input_sentence):
    test_sentence_tokenized[0, t, tokenized_encoder_intvalue[char]] = 1.
print(input_sentence)
print(decode_sequence(test_sentence_tokenized))