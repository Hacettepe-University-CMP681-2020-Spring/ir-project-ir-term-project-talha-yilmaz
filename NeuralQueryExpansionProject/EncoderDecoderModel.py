import numpy as np
from pickle import load, dump
from keras_preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from attention import AttentionLayer
from sklearn.model_selection import train_test_split

querytweets = load(open('data/querytweets.pkl', 'rb'))
print('Loaded Query Tweets Length: %d' % len(querytweets))
print(type(querytweets))

max_tweet_len=50
max_query_len=10
batch_size = 256
epochs = 200

cleaned_tweet=[]
cleaned_query=[]

for querytweet in querytweets:
    if len(querytweet['Tweet'].split()) <= max_tweet_len and len(querytweet['Query'].split()) <= max_query_len:
        cleaned_tweet.append(querytweet['Tweet'])
        cleaned_query.append(querytweet['Query'])


x_train, x_validation, y_train, y_validation = train_test_split(cleaned_tweet, cleaned_query, test_size=0.1, random_state=0, shuffle=True)

#prepare a tokenizer for tweets on training data
x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_train))

#convert text sequences to one hot encoding sequences
x_tr_seq = x_tokenizer.texts_to_sequences(x_train)
x_val_seq = x_tokenizer.texts_to_sequences(x_validation)

#post-padding text sequences
x_train = pad_sequences(x_tr_seq, maxlen=max_tweet_len, padding='post')
x_validation = pad_sequences(x_val_seq, maxlen=max_tweet_len, padding='post')

#size of vocabulary ( +1 for padding token)
x_vocab_size = len(x_tokenizer.word_counts.items()) + 1

print("Size of vocabulary in X = {}".format(x_vocab_size))


#prepare a tokenizer for queries on training data
y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_train))

#convert summary queries to one hot encoding sequences
y_tr_seq = y_tokenizer.texts_to_sequences(y_train)
y_val_seq = y_tokenizer.texts_to_sequences(y_validation)

#post-padding summary sequences
y_train = pad_sequences(y_tr_seq, maxlen=max_query_len, padding='post')
y_validation = pad_sequences(y_val_seq, maxlen=max_query_len, padding='post')

#size of vocabulary
y_vocab_size = len(y_tokenizer.word_counts.items()) + 1

print("Size of vocabulary in Y = {}".format(y_vocab_size))


latent_dim = 300
embedding_dim = 200

# Encoder
encoder_inputs = Input(shape=(max_tweet_len,))

#embedding layer
enc_emb = Embedding(x_vocab_size, embedding_dim, trainable=True)(encoder_inputs)

#encoder lstm layer 1
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

#encoder lstm layer 2
encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

#encoder lstm layer 3
encoder_lstm3 = LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

initial_state = [state_h, state_c]

# Set up the decoder
decoder_inputs = Input(shape=(None,))

#decoder embedding layer
dec_emb_layer = Embedding(y_vocab_size, embedding_dim, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.4)
decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=initial_state)

attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#dense layer
decoder_dense = TimeDistributed(Dense(y_vocab_size, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)
model.fit([x_train, y_train[:, :-1]], y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:, 1:],
                    epochs=epochs,
                    callbacks=[es],
                    batch_size=batch_size,
                    validation_data=([x_validation, y_validation[:, :-1]], y_validation.reshape(y_validation.shape[0], y_validation.shape[1], 1)[:, 1:]))


reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index

# Encode the input sequence to get the feature vector
encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

# Decoder setup, below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_tweet_len, latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2 = dec_emb_layer(decoder_inputs)

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

#attention inference
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_inf_concat)

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['querybaslangici']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if sampled_token != 'querybitisi':
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find stop word.
        if sampled_token == 'querybitisi' or len(decoded_sentence.split()) >= (max_query_len - 1):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


def seq2query(input_seq):
    newString=''
    for i in input_seq:
        if (i!=0 and i!=target_word_index['querybaslangici']) and i!=target_word_index['querybitisi']:
            newString=newString+reverse_target_word_index[i]+' '
    return newString


def seq2tweet(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString


predicted_query_tweets = list()

for i in range(len(querytweets)-(len(querytweets)//10)+1):
    tweet = seq2tweet(x_train[i])
    original_query = seq2query(y_train[i])
    predicted_query = decode_sequence(x_train[i].reshape(1,max_tweet_len))

    predicted_query_tweets.append({'Tweet': tweet,
                                   'Original_Query': original_query,
                                   'Predicted_Query': predicted_query})

dump(predicted_query_tweets, open('data/predictedquerytweets.pkl', 'wb'))
