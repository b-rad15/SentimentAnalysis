import warnings
import helper
import time
from ModelProcessing import ModelProcessing
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import nltk
import tensorflow as tf
import numpy as np


warnings.filterwarnings('ignore')

model_object = ModelProcessing(yaml_file="model-cuda.yaml")
model = model_object.model
tokenizer = model_object.tokenizer

encoding = model_object.output_dict
encoding = {v: k for k, v in encoding.items()}

while (message := input("Enter a message (enter \"quit\" to quit): ")) != "quit":
    padded_message = pad_sequences(tokenizer.texts_to_sequences([message]), maxlen=model_object.max_seq_len)
    pred = model.predict(padded_message)
    print('Emotion:', encoding[np.argmax(pred)])
    helper.print_confidences(pred[0], encoding)

