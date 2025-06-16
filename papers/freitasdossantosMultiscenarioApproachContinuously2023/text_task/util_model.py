import pathlib

import numpy as np
import tensorflow as tf
import timeit
import keras.metrics
import torch

from sklearn.metrics import classification_report
from keras import Input
from keras.layers import Dense
from keras.optimizers.optimizer_v2.adam import Adam
from keras.losses import BinaryCrossentropy, BinaryFocalCrossentropy
from transformers import RobertaConfig, RobertaForSequenceClassification, DistilBertConfig, DistilBertForSequenceClassification, RobertaTokenizer, AutoTokenizer

ROBERTA_MODEL_NAME = "roberta-base"
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"


def run_build(bert_model, number_of_output_nodes, activation_function="sigmoid", max_input_len=64):
    input_ids = Input(shape=(max_input_len,), dtype=tf.int32, name="input_ids")
    input_mask = Input(shape=(max_input_len,), dtype=tf.int32, name="attention_mask")

    # bert_model[0] contains the last Bert hidden state. bert_model[1] contains the pooled_output.
    # embeddings = bert_model.bert(input_ids, attention_mask=input_mask)[1]
    embeddings = bert_model.roberta(input_ids, attention_mask=input_mask)[1]

    out_dropout = tf.keras.layers.Dropout(0.1)(embeddings)
    final_output = Dense(number_of_output_nodes, activation=activation_function)(out_dropout)

    model = tf.keras.models.Model(inputs=[input_ids, input_mask], outputs=final_output)
    model.layers[2].trainable = True

    return model


def run_build_distil(distilbert_model, number_of_output_nodes, activation_function="sigmoid", max_input_len=64):
    input_ids = Input(shape=(max_input_len,), dtype=tf.int32, name="input_ids")
    input_mask = Input(shape=(max_input_len,), dtype=tf.int32, name="attention_mask")

    # bert_model[0] contains the last Bert hidden state. bert_model[1] contains the pooled_output.
    embeddings = distilbert_model(input_ids, attention_mask=input_mask)[0]

    embeddings = embeddings[:, 0, :]

    # out_dropout = tf.keras.layers.Dropout(0.1)(embeddings)
    final_output = Dense(number_of_output_nodes, activation=activation_function)(embeddings)

    model = tf.keras.models.Model(inputs=[input_ids, input_mask], outputs=final_output)
    model.layers[2].trainable = True

    return model


def define_model_class(model_name, number_of_labels):
    if model_name == ROBERTA_MODEL_NAME:
        roberta_config = RobertaConfig.from_pretrained(ROBERTA_MODEL_NAME, num_labels=number_of_labels)
        return RobertaForSequenceClassification.from_pretrained(ROBERTA_MODEL_NAME, config=roberta_config)
    elif model_name == DISTILBERT_MODEL_NAME:
        distil_config = DistilBertConfig(dropout=0.1, attention_dropout=0.1, output_hidden_states=True, num_labels=number_of_labels)
        return DistilBertForSequenceClassification.from_pretrained(model_name, config=distil_config)


def define_tokenizer(model_name, max_input_len=64):
    return AutoTokenizer.from_pretrained(model_name, add_special_tokens=True,
                                         max_length=max_input_len,
                                         truncation=True,
                                         padding="max_length",
                                         return_tensors="pt",
                                         return_token_type_ids=False,
                                         return_attention_mask=True,
                                         verbose=True)


def tokenize_data(dataset, tokenizer, max_input_len=64):
    # Tokenizing the text data
    tokenized_data = tokenizer(text=dataset["NEW_FORMATTED_TEXT"].tolist(),
                               add_special_tokens=True,
                               max_length=max_input_len,
                               truncation=True,
                               padding="max_length",
                               return_tensors="tf",
                               return_token_type_ids=False,
                               return_attention_mask=True,
                               verbose=True)

    return tokenized_data


def run_compile(model):
    optimizer = Adam(learning_rate=1e-4,  # 5e-5
                     epsilon=1e-08,
                     decay=0.01,
                     clipnorm=1)
    # optimizer = SGD(learning_rate=0.01)

    loss = BinaryFocalCrossentropy(from_logits=False)
    metric = keras.metrics.BinaryAccuracy("binary_accuracy")

    start_time = timeit.default_timer()
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metric)
    finish_time = timeit.default_timer()
    compile_time = finish_time - start_time

    return model, compile_time


def run_fit(model, tokenized_training_data, y_train, epochs, batch_size, tokenized_validation_data=None, y_validation=None):
    if tokenized_validation_data is None:
        start_time = timeit.default_timer()
        model.fit(x={"input_ids": tokenized_training_data["input_ids"],
                     "attention_mask": tokenized_training_data["attention_mask"]},
                  y=y_train,
                  epochs=epochs,
                  batch_size=batch_size)
    else:
        start_time = timeit.default_timer()
        model.fit(x={"input_ids": tokenized_training_data["input_ids"],
                     "attention_mask": tokenized_training_data["attention_mask"]},
                  y=y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=({"input_ids": tokenized_validation_data["input_ids"],
                                    "attention_mask": tokenized_validation_data["attention_mask"]},
                                   y_validation))

    finish_time = timeit.default_timer()
    training_time = finish_time - start_time

    return model, training_time


def get_individual_predictions(dataset, predictions):
    individual_predictions = []
    current_index = 0
    for edit_index, edit in dataset.iterrows():
        edit_text = edit["NEW_FORMATTED_TEXT"]

        individual_prediction_info = {
            "edit_text": edit["TEXT"],
            "formatted_text": edit_text,
            "real_label": edit["NEWMULTILABEL"],
            "predicted_label": predictions[current_index],
            "edit_url": edit["URL"]
        }

        '''if set(predictions[current_index]) != set(edit["NEWMULTILABEL"]):
            print(edit_text)
            print(set(edit["NEWMULTILABEL"]))
            print(set(predictions[current_index]))
            print("URL: " + edit["URL"])
            print("\n")
        '''

        individual_predictions.append(individual_prediction_info)
        current_index += 1

    return individual_predictions


def get_individual_predictions_bin(dataset, predictions, label):
    individual_predictions = []
    current_index = 0
    for edit_index, edit in dataset.iterrows():
        edit_text = edit["NEW_FORMATTED_TEXT"]

        individual_prediction_info = {
            "edit_text": edit["TEXT"],
            "formatted_text": edit_text,
            "real_label": edit[label],
            "predicted_label": predictions[current_index],
            "edit_url": edit["URL"]
        }

        individual_predictions.append(individual_prediction_info)
        current_index += 1

    return individual_predictions


def run_evaluate(model, tokenized_test_data, y_test):
    y_probabilities = model.predict(x={"input_ids": tokenized_test_data["input_ids"],
                                       "attention_mask": tokenized_test_data["attention_mask"]})

    y_predicted = np.where(y_probabilities > 0.5, 1, 0)
    calculated_classification_report = classification_report(y_test, y_predicted, output_dict=True, zero_division=0)

    return calculated_classification_report, y_predicted


