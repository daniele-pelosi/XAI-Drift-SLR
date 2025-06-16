import pathlib
import pickle
import timeit
import torch
from transformers import AutoTokenizer, TFBertModel, BertConfig, BertModel, RobertaTokenizer, RobertaConfig, RobertaModel, \
    TFDistilBertModel, DistilBertConfig, TFRobertaModel, RobertaForSequenceClassification, TFRobertaForSequenceClassification, DistilBertForSequenceClassification

from transformers_interpret import SequenceClassificationExplainer

import util_interpretability
import util_model
import util_model_pt

CUDA_VISIBLE_DEVICES = ""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BERT_MODEL_NAME = "bert-base-uncased"
ROBERTA_MODEL_NAME = "roberta-base"
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
MAX_INPUT_LEN = 64


def execute_interpretability(model, tokenizer, texts, real_labels, current_step):
    cls_explainer = SequenceClassificationExplainer(model, tokenizer)

    index = 0
    for text_to_explain in texts:
        word_attributions = cls_explainer(text_to_explain)
        print(text_to_explain)
        print(word_attributions)
        print("Real Label: " + str(real_labels.iloc[index]))
        print("Predicted Label: " + str(cls_explainer.predicted_class_index))

        file_name = "IG_" + str(current_step) + "_" + str(index) + ".html"
        location_to_save = "interpretability_files_test/" + file_name
        cls_explainer.visualize(location_to_save)

        index += 1


def execute_interpretability_from_disk(dataset_directory, dataset_file_name, number_k_datasets, model_directory,
                                       k_folder, interpretability_step, testing_data_index):
    base_directory = str(pathlib.Path(__file__).parent)
    complete_dataset_directory = base_directory + dataset_directory + dataset_file_name

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME,
                                              add_special_tokens=True,
                                              max_length=MAX_INPUT_LEN,
                                              truncation=True,
                                              padding="max_length",
                                              return_tensors="pt",
                                              return_token_type_ids=False,
                                              return_attention_mask=True,
                                              verbose=True)

    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_NAME, add_special_tokens=True,
                                                 max_length=MAX_INPUT_LEN,
                                                 truncation=True,
                                                 padding="max_length",
                                                 return_tensors="pt",
                                                 return_token_type_ids=False,
                                                 return_attention_mask=True,
                                                 verbose=True)

    complete_model_directory = base_directory + model_directory + str(k_folder) + "/trained_model/"
    '''bert_config = BertConfig.from_pretrained(BERT_MODEL_NAME, num_labels=2)
    model = BertModel.from_pretrained(BERT_MODEL_NAME, config=bert_config)
    loaded_model = keras.models.load_model(complete_model_directory)'''
    roberta_config = RobertaConfig.from_pretrained(ROBERTA_MODEL_NAME, num_labels=2)
    model = RobertaModel.from_pretrained(ROBERTA_MODEL_NAME, config=roberta_config)

    # model = util_model.run_build(model, 1, "sigmoid")
    # model, current_compile_time = util_model.run_compile(model)

    texts_file_location = "interpretability_files/x_test_" + str(testing_data_index) + ".pickle"
    real_labels_file_location = "interpretability_files/y_test_" + str(testing_data_index) + ".pickle"
    texts = pickle.load(open(texts_file_location, "rb"))
    real_labels = pickle.load(open(real_labels_file_location, "rb"))

    cls_explainer = SequenceClassificationExplainer(model, tokenizer)

    index = 0
    for text_to_explain in texts:
        text_to_explain = text_to_explain.replace("\u0122", "")

        start_time = timeit.default_timer()
        word_attributions = cls_explainer(text_to_explain)
        finish_time = timeit.default_timer()
        interpretability_time = finish_time - start_time

        if isinstance(real_labels, list):
            true_class = real_labels[index]
        else:
            true_class = real_labels.iloc[index]
        predicted_class = cls_explainer.predicted_class_index

        print(word_attributions)
        print("Real Label: " + str(true_class))
        print("Predicted Label: " + str(predicted_class))
        print("Interpretability Time (sec): " + str(interpretability_time))

        prediction_status = "_CORRECT"
        if true_class != predicted_class:
            prediction_status = "_WRONG"

        file_name = "IG_" + str(interpretability_step) + "_" + str(index) + prediction_status + ".html"

        location_to_save = "interpretability_files/" + file_name
        cls_explainer.visualize(location_to_save, true_class=true_class)

        index += 1


def execute_interpretability_from_disk_all(dataset_directory, model_directory, dataset_file_name, k_folder, interpretability_directory,
                                           interpretability_step, model_name):
    base_directory = str(pathlib.Path(__file__).parent)
    complete_dataset_directory = base_directory + dataset_directory + dataset_file_name

    tokenizer = define_tokenizer(model_name)

    complete_model_directory = base_directory + model_directory + str(k_folder) + "/trained_model/"
    # loaded_model = keras.models.load_model(complete_model_directory)
    roberta_config = RobertaConfig.from_pretrained(ROBERTA_MODEL_NAME, num_labels=2)
    # loaded_model = TFRobertaModel.from_pretrained(ROBERTA_MODEL_NAME, config=roberta_config)
    loaded_model = TFRobertaForSequenceClassification.from_pretrained(ROBERTA_MODEL_NAME, config=roberta_config)
    # loaded_model = util_model.run_build(loaded_model, 1, "sigmoid")

    distil_config = DistilBertConfig(dropout=0.1, attention_dropout=0.1, output_hidden_states=True, num_labels=2)
    loaded_model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_MODEL_NAME, config=distil_config)

    test_dataset_location = interpretability_directory + "test_violation.pickle"
    test_dataset = pickle.load(open(test_dataset_location, "rb"))

    cls_explainer = SequenceClassificationExplainer(loaded_model, tokenizer)

    interpretability_results = []
    index = 0
    for index, edit in test_dataset.iterrows():
        text_to_explain = edit["NEW_FORMATTED_TEXT"]
        true_class = edit["VANDALISM"]

        text_to_explain = text_to_explain.replace("\u0122", "")
        print(text_to_explain)

        if len(text_to_explain.split()) > 300:
            text_to_explain = " ".join(text_to_explain.split()[0:300])

        start_time = timeit.default_timer()
        tokens_scores = cls_explainer(text_to_explain)
        tokenized_text_to_explain = tokenizer.tokenize(text_to_explain)  # Use this to know when a word is split into different tokens.
        finish_time = timeit.default_timer()
        interpretability_time = finish_time - start_time

        words_scores = util_interpretability.calculate_words_scores(tokens_scores, tokenized_text_to_explain)

        predicted_class = cls_explainer.predicted_class_index

        print(words_scores)
        print("Real Label: " + str(true_class))
        print("Predicted Label: " + str(predicted_class))
        print("Interpretability Time (sec): " + str(interpretability_time) + "\n")

        prediction_status = "_CORRECT"
        correct_prediction = True
        if true_class != predicted_class:
            prediction_status = "_WRONG"
            correct_prediction = False

        current_interpretability = {
            "tokens_scores": tokens_scores,
            "words_scores": words_scores,
            "explained_text": text_to_explain,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "prediction_score": cls_explainer.pred_probs,
            "corrected_prediction": correct_prediction
        }

        interpretability_results.append(current_interpretability)
        file_name = "IG_violation_" + str(index) + prediction_status + ".html"

        location_to_save = interpretability_directory + file_name
        cls_explainer.visualize(location_to_save, true_class=true_class)

        index += 1

    interpretability_results_file = "interpret_results_" + str(interpretability_step) + ".pickle"
    interpretability_to_save = interpretability_directory + interpretability_results_file
    pickle.dump(interpretability_results, open(interpretability_to_save, 'wb'))


def execute_interpretability_complete_data(dataset_directory, model_directory, dataset_file_name, interpretability_directory,
                                           interpretability_step, interpretability_model_directory, model_name, number_of_labels):
    base_directory = str(pathlib.Path(__file__).parent)
    # complete_dataset_directory = base_directory + model_directory + dataset_file_name

    tokenizer = define_tokenizer(model_name)
    model_class = util_model_pt.define_model_class(model_name, number_of_labels, "single_label_classification")
    loaded_model = util_model_pt.load_model_interpretability(model_class, model_directory)

    test_dataset_location = base_directory + interpretability_directory + dataset_file_name
    test_dataset = pickle.load(open(test_dataset_location, "rb"))

    cls_explainer = SequenceClassificationExplainer(loaded_model, tokenizer)

    interpretability_results = []
    index = 0
    for edit_index, edit in test_dataset.iterrows():
        if index > -1:
            text_to_explain = edit["NEW_FORMATTED_TEXT"]
            true_class = edit["VANDALISM"]

            text_to_explain = text_to_explain.replace("\u0122", "")
            print(text_to_explain)

            if len(text_to_explain.split()) > 300:
                text_to_explain = " ".join(text_to_explain.split()[0:300])

            start_time = timeit.default_timer()
            tokens_scores = cls_explainer(text_to_explain)
            tokenized_text_to_explain = tokenizer.tokenize(text_to_explain)  # Use this to know when a word is split into different tokens.
            finish_time = timeit.default_timer()
            interpretability_time = finish_time - start_time

            if model_name == ROBERTA_MODEL_NAME:
                words_scores = util_interpretability.calculate_words_scores(tokens_scores, tokenized_text_to_explain)
            else:
                words_scores = util_interpretability.calculate_words_scores_distil(tokens_scores, tokenized_text_to_explain)

            predicted_class = cls_explainer.predicted_class_index

            print(words_scores)
            print("Real Label: " + str(true_class))
            print("Predicted Label: " + str(predicted_class))
            print("Interpretability Time (sec): " + str(interpretability_time) + "\n")

            prediction_status = "_CORRECT"
            correct_prediction = True
            if true_class != predicted_class:
                prediction_status = "_WRONG"
                correct_prediction = False

            current_interpretability = {
                "tokens_scores": tokens_scores,
                "words_scores": words_scores,
                "explained_text": text_to_explain,
                "true_class": true_class,
                "predicted_class": predicted_class,
                "prediction_score": cls_explainer.pred_probs,
                "corrected_prediction": correct_prediction
            }

            interpretability_results.append(current_interpretability)
            file_name = "IG_violation_" + str(edit_index) + prediction_status + ".html"

            location_to_save = base_directory + interpretability_directory + interpretability_model_directory + file_name
            cls_explainer.visualize(location_to_save, true_class=true_class)

        index += 1

    interpretability_results_file = "interpret_results_" + str(interpretability_step) + ".pickle"
    interpretability_to_save = base_directory + interpretability_directory + interpretability_model_directory + str(interpretability_step) + "/" + interpretability_results_file
    pickle.dump(interpretability_results, open(interpretability_to_save, 'wb'))


def execute_interpretability_complete_data_multi(dataset_directory, model_directory, dataset_file_name, interpretability_directory,
                                                 interpretability_step, interpretability_model_directory, model_name, number_of_labels):
    base_directory = str(pathlib.Path(__file__).parent)
    complete_dataset_directory = base_directory + model_directory + dataset_file_name

    tokenizer = define_tokenizer(model_name)
    model_class = util_model.define_model_class(model_name, number_of_labels)
    loaded_model = util_model.load_model_interpretability(model_class, model_directory)

    test_dataset_location = base_directory + interpretability_directory + "test_violation.pickle"
    test_dataset = pickle.load(open(test_dataset_location, "rb"))

    cls_explainer = SequenceClassificationExplainer(loaded_model, tokenizer)

    interpretability_results = []
    index = 0
    for edit_index, edit in test_dataset.iterrows():
        text_to_explain = edit["NEW_FORMATTED_TEXT"]
        true_class = edit["VANDALISM"]

        text_to_explain = text_to_explain.replace("\u0122", "")
        print(text_to_explain)

        if len(text_to_explain.split()) > 300:
            text_to_explain = " ".join(text_to_explain.split()[0:300])

        start_time = timeit.default_timer()
        tokens_scores = cls_explainer(text_to_explain)
        tokenized_text_to_explain = tokenizer.tokenize(text_to_explain)  # Use this to know when a word is split into different tokens.
        finish_time = timeit.default_timer()
        interpretability_time = finish_time - start_time

        words_scores = util_interpretability.calculate_words_scores(tokens_scores, tokenized_text_to_explain)

        predicted_class = cls_explainer.predicted_class_index

        print(words_scores)
        print("Real Label: " + str(true_class))
        print("Predicted Label: " + str(predicted_class))
        print("Interpretability Time (sec): " + str(interpretability_time) + "\n")

        prediction_status = "_CORRECT"
        correct_prediction = True
        if true_class != predicted_class:
            prediction_status = "_WRONG"
            correct_prediction = False

        current_interpretability = {
            "tokens_scores": tokens_scores,
            "words_scores": words_scores,
            "explained_text": text_to_explain,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "prediction_score": cls_explainer.pred_probs,
            "corrected_prediction": correct_prediction
        }

        interpretability_results.append(current_interpretability)
        file_name = "IG_violation_" + str(edit_index) + prediction_status + ".html"

        location_to_save = base_directory + interpretability_directory + interpretability_model_directory + file_name
        cls_explainer.visualize(location_to_save, true_class=true_class)

        index += 1

    interpretability_results_file = "interpret_results_" + str(interpretability_step) + ".pickle"
    interpretability_to_save = interpretability_directory + interpretability_model_directory + str(interpretability_step) + "/" + interpretability_results_file
    pickle.dump(interpretability_results, open(interpretability_to_save, 'wb'))


def define_tokenizer(model_name):
    if model_name == ROBERTA_MODEL_NAME:
        return RobertaTokenizer.from_pretrained(ROBERTA_MODEL_NAME, add_special_tokens=True,
                                                max_length=MAX_INPUT_LEN,
                                                truncation=True,
                                                padding="max_length",
                                                return_tensors="pt",
                                                return_token_type_ids=False,
                                                return_attention_mask=True,
                                                verbose=True)
    else:
        return AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME, add_special_tokens=True,
                                             max_length=MAX_INPUT_LEN,
                                             truncation=True,
                                             padding="max_length",
                                             return_tensors="pt",
                                             return_token_type_ids=False,
                                             return_attention_mask=True,
                                             verbose=True)
