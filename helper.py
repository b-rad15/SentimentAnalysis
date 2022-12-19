import pandas as pd
from keras.preprocessing.text import Tokenizer
import re
from nltk.tokenize import word_tokenize
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, GRU, Dense
import pickle
import tensorflow as tf


def prep_data(filename: str):
    # training
    x_data = []
    y_data = []
    with open(filename, 'r') as f:
        for i in f:
            line_split = i.split(';')
            y_data.append(line_split[1].strip())
            x_data.append(line_split[0])
    data = pd.DataFrame({'Text': x_data, 'Emotion': y_data})
    return data


def make_tokenizer(texts):
    # tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer


regex1 = re.compile(r"(#[\d\w\.]+)")
regex2 = re.compile(r"(@[\d\w\.]+)")


def clean_text(data):
    data = regex1.sub('', data)
    data = regex2.sub('', data)
    data = word_tokenize(data)
    return data


# pre-trained word vectors
def create_embedding_matrix(word_index, embedding_dim, filepath: str = 'models/crawl-300d-2M.vec'):
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(filepath, 'r', encoding='utf-8') as vec_file:
        for line in vec_file:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix


def create_model(cuda_optimized_gru: bool, tokenizer: Tokenizer, embed_num_dims: int, embed_filepath: str, max_seq_len: int,
                 num_classes: int):
    embedd_matrix = create_embedding_matrix(tokenizer.word_index, embed_num_dims, embed_filepath)

    # adding layers
    embedded_layer = Embedding(len(tokenizer.word_index) + 1, embed_num_dims, input_length=max_seq_len,
                               weights=[embedd_matrix],
                               trainable=False)
    gru_output_size = 150
    bidirectional = True
    model = Sequential()
    model.add(embedded_layer)
    # Cudnn Compatibility using reference https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU#used-in-the-notebooks
    gru_layer = GRU(units=gru_output_size, dropout=0.2, recurrent_dropout=0.2) if not cuda_optimized_gru \
        else GRU(units=gru_output_size, activation="tanh", recurrent_activation='sigmoid', recurrent_dropout=0,
                 unroll=False, use_bias=True, reset_after=True, dropout=0.2)
    model.add(Bidirectional(gru_layer))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def pickle_save_object(object_to_pickle: object, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(object_to_pickle, f, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_read_object(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_model(model_to_save: object, foldername: str, overwrite: bool = True):
    # save the model for later use
    tf.keras.models.save_model(model_to_save, foldername, overwrite=overwrite, include_optimizer=True, save_format=None,
                               signatures=None, options=None)


def print_confidences(prediction, encoding_dict: dict):
    for k, v in encoding_dict.items():
        print(f"{k}: {prediction[v]:.1%}")
