import tensorflow as tf
import pickle
import yaml
import helper
from keras_preprocessing.sequence import pad_sequences

tf.config.experimental.set_memory_growth = True


class ModelProcessing:
    def __init__(self, model=None, tokenizer=None, max_seq_len: int = None, output_dict: dict = None,
                 yaml_file: str = None):
        self._tokenizer = None
        self._model = None
        if yaml_file:
            # Constructor to read from a YAML file
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                self.model_file = data['model_file']
                self.tokenizer_file = data['tokenizer_file']
                self.output_dict: dict = data['output_dict']
                self.max_seq_len: int = data['max_seq_len']
                self._model = tf.keras.models.load_model(self.model_file)
                with open(self.tokenizer_file, 'rb') as tokenizer_fileread:
                    self._tokenizer = pickle.load(tokenizer_fileread)
        elif model and tokenizer and output_dict and max_seq_len:
            # Constructor that takes the model, tokenizer, and output dictionary as input
            self.model = model
            self.tokenizer = tokenizer
            self.output_dict: dict = output_dict
            self.max_seq_len: int = max_seq_len
            self.model_file: str = None
            self.tokenizer_file: str = None
        else:
            raise "Must specify yaml_file or all of model and tokenizer and output_dict"

    def write_to_yaml(self, yaml_file: str, model_file: str = None, tokenizer_file: str = None):
        error = ""
        if not hasattr(self, '_model') or self._model is None:
            error += "No savepath specified for model, must specify a path to save to\n"
        if not hasattr(self, '_tokenizer') or self._tokenizer is None:
            error += "No savepath specified for tokenizer, must specify a path to save to\n"
        if len(error) > 0:
            # Todo: save to random, untaken, temporary paths so models not lost
            raise error
        if tokenizer_file:
            self.tokenizer_file = tokenizer_file
        if model_file:
            self.model_file = model_file
        # Function to write out to a YAML file
        helper.pickle_save_object(self._tokenizer, self.tokenizer_file)
        helper.save_model(self.model, self.model_file)
        data = {'model_file': self.model_file, 'tokenizer_file': self.tokenizer_file, 'output_dict': self.output_dict,
                "max_seq_len": self.max_seq_len}
        with open(yaml_file, 'w') as f:
            yaml.safe_dump(data, f)

    @tf.function(reduce_retracing=True)
    def tokenize_and_sequence(self, messages: list):
        return pad_sequences(self.tokenize_messages(messages), maxlen=self.max_seq_len)

    def tokenize_messages(self, messages: list):
        return self.tokenizer.texts_to_sequences(messages)

    @tf.function(reduce_retracing=True)
    def predict_padded(self, padded_messages):
        return self.model(padded_messages)

    def predict_messages(self, messages: list):
        return self.predict_padded(self.tokenize_and_sequence(messages))

    def predict_message(self, message: str):
        return self.predict_messages([message])[0]

    @property
    def model(self):
        # Load model from file if not already loaded
        if not hasattr(self, '_model') or self._model is None:
            self._model = tf.keras.models.load_model(self.model_file)
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def tokenizer(self):
        # load tokenizer from file if not already loaded
        if not hasattr(self, '_tokenizer') or self._tokenizer is None:
            with open(self.tokenizer_file, 'rb') as f:
                self._tokenizer = pickle.load(f)
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value
