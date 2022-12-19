import pandas as pd
import numpy as np
import re
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import nltk
from nltk.tokenize import word_tokenize
import warnings
import helper
import time
from ModelProcessing import ModelProcessing


nltk.download('punkt')
warnings.filterwarnings('ignore')

data_test = helper.prep_data("test.txt")

model_object = ModelProcessing(yaml_file="model-cuda.yaml")

# tokenization
tokenizer = model_object.tokenizer
texts_test = [' '.join(helper.clean_text(text)) for text in data_test.Text]
sequence_test = tokenizer.texts_to_sequences(texts_test)

# categorization
encoding = model_object.output_dict
max_seq_len = model_object.max_seq_len
x_test_pad = pad_sequences(sequence_test, maxlen=max_seq_len)
y_test = [encoding[x] for x in data_test.Emotion]
y_test_categorical = to_categorical(y_test)

total_elements = len(data_test.Text)

# %%

model = model_object.model
start = time.perf_counter()
pred = model.predict(x_test_pad)
end = time.perf_counter()
print(f"Model {model_object.model_file} took {end - start} seconds to execute")
# %%
predicted_indices = np.argmax(pred, axis=1)
predicted_confidences = np.max(pred, axis=1)
was_correct = predicted_indices == [encoding[x] for x in data_test.Emotion]
correct_confidences = np.extract(was_correct, predicted_confidences)
incorrect_confidences = np.extract(~was_correct, predicted_confidences)
correct_total = len(correct_confidences)
incorrect_total = len(incorrect_confidences)
correct_mean = np.mean(correct_confidences)
incorrect_mean = np.mean(incorrect_confidences)
correct_stdev = np.std(correct_confidences)
incorrect_stdev = np.std(incorrect_confidences)
z_score_cutoff = (correct_mean - incorrect_mean) / (correct_stdev + incorrect_stdev)
cutoff = correct_mean - correct_stdev * z_score_cutoff  # values greater than this are "certain', values less are "unsure"

print(f"{correct_total} correct with average confidence {correct_mean}")
print(f"{incorrect_total} incorrect with average confidence {incorrect_mean}")
print(f"Accuracy: {correct_total / total_elements:0>6.2%}")
print("Done")
