import pandas as pd
import numpy as np
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
import urllib.request
import zipfile
import os
import nltk
import warnings
import helper
from ModelProcessing import ModelProcessing

nltk.download('punkt')
warnings.filterwarnings('ignore')

data_train = helper.prep_data("train.txt")
data_test = helper.prep_data("test.txt")
data_val = helper.prep_data("val.txt")
class_names = ['anger', 'sadness', 'fear', 'joy', 'surprise', 'love']

data = data_train.append(data_test, ignore_index=True)
data = data.append(data_val, ignore_index=True)

texts = [' '.join(helper.clean_text(text)) for text in data.Text]
texts_train = [' '.join(helper.clean_text(text)) for text in data_train.Text]
texts_val = [' '.join(helper.clean_text(text)) for text in data_val.Text]

tokenizer = helper.make_tokenizer(texts)
sequence_train = tokenizer.texts_to_sequences(texts_train)
sequence_val = tokenizer.texts_to_sequences(texts_val)
index_of_words = tokenizer.word_index
vocab_size = len(index_of_words) + 1

# categorization
embed_num_dims = 300  # derived from vector of words model
max_seq_len = 500
X_train_pad = pad_sequences(sequence_train, maxlen=max_seq_len)
X_val_pad = pad_sequences(sequence_val, maxlen=max_seq_len)
encoding = dict(zip(class_names, range(len(class_names))))
y_train = [encoding[x] for x in data_train.Emotion]
y_val = [encoding[x] for x in data_val.Emotion]
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

model = helper.create_model(cuda_optimized_gru=True, tokenizer=tokenizer, embed_num_dims=embed_num_dims, embed_filepath='models/crawl-300d-2M.vec', max_seq_len=max_seq_len, num_classes=len(class_names))

# training
batch_size = 512
epochs = 20
hist = model.fit(X_train_pad, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val_pad, y_val))

# testing model
message = 'I love machine learning using python.'
seq = tokenizer.texts_to_sequences([message])
padded = pad_sequences(seq, maxlen=max_seq_len)
pred = model.predict(padded)
print('Message:' + str(message))
print('Emotion:', class_names[np.argmax(pred)])

model_object = ModelProcessing(model=model, tokenizer=tokenizer, max_seq_len=max_seq_len, output_dict=encoding)
model_object.write_to_yaml(yaml_file="model-cuda.yaml", model_file="model-cuda", tokenizer_file="tokenizer.pkl")
