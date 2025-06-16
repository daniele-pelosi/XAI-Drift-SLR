import pathlib
import pickle
import timeit
import util_interpretability
import util_model_pt
import torch

from transformers import AutoTokenizer, RobertaTokenizer
from transformers_interpret.explainers.text import MultiLabelClassificationExplainer

CUDA_VISIBLE_DEVICES = ""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BERT_MODEL_NAME = "bert-base-uncased"
ROBERTA_MODEL_NAME = "roberta-base"
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
MAX_INPUT_LEN = 64


def execute_interpretability_complete_data(model_directory, dataset_file_name, interpretability_directory,
                                           interpretability_step, interpretability_model_directory, model_name, number_of_labels):
    base_directory = str(pathlib.Path(__file__).parent)

    tokenizer = define_tokenizer(model_name)
    id2label = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
    label2id = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    model_class = util_model_pt.define_model_class(model_name, number_of_labels, "multi_label_classification", id2label, label2id)
    loaded_model = util_model_pt.load_model_interpretability(model_class, model_directory)

    test_dataset_location = base_directory + interpretability_directory + dataset_file_name
    test_dataset = pickle.load(open(test_dataset_location, "rb"))

    cls_explainer = MultiLabelClassificationExplainer(loaded_model, tokenizer)

    interpretability_results = {}
    for label in range(number_of_labels):
        interpretability_results[str(label)] = []

    index = 0
    for edit_index, edit in test_dataset.iterrows():
        if edit_index == 15375 or edit_index == 15274 or edit_index == 15412 or edit_index == 15085 or edit_index == 15633 or edit_index == 15647:
            text_to_explain = edit["NEW_FORMATTED_TEXT"]
            true_class = edit["NEWMULTILABEL"]

            text_to_explain = text_to_explain.replace("\u0122", "")
            print(text_to_explain)

            if len(text_to_explain.split()) > 300:
                text_to_explain = " ".join(text_to_explain.split()[0:300])

            start_time = timeit.default_timer()
            all_labels_tokens_scores = cls_explainer(text_to_explain)
            tokenized_text_to_explain = tokenizer.tokenize(text_to_explain)  # Use this to know when a word is split into different tokens.
            finish_time = timeit.default_timer()
            interpretability_time = finish_time - start_time

            pred_probs = cls_explainer.pred_probs
            predicted_classes = []
            for index in range(len(pred_probs)):
                predicted_classes.append(int((pred_probs[index].item() > 0.5)))

            for label in range(number_of_labels):
                if model_name == ROBERTA_MODEL_NAME:
                    tokens_scores = all_labels_tokens_scores["LABEL_" + str(label)]
                    words_scores = util_interpretability.calculate_words_scores(tokens_scores, tokenized_text_to_explain)
                else:
                    tokens_scores = all_labels_tokens_scores[str(label)]
                    words_scores = util_interpretability.calculate_words_scores_distil(tokens_scores, tokenized_text_to_explain)

                prediction_status = "_CORRECT"
                correct_prediction = True
                if true_class[label] != predicted_classes[label]:
                    prediction_status = "_WRONG"
                    correct_prediction = False

                current_interpretability = {
                    "tokens_scores": tokens_scores,
                    "words_scores": words_scores,
                    "explained_text": text_to_explain,
                    "current_class": label,
                    "true_value": true_class[label],
                    "predicted_value": predicted_classes[label],
                    "multilabel": predicted_classes,
                    "corrected_prediction": correct_prediction
                }

                interpretability_results[str(label)].append(current_interpretability)

            file_name = "IG_violation_" + str(edit_index) + ".html"

            location_to_save = base_directory + interpretability_directory + interpretability_model_directory + file_name
            cls_explainer.visualize(location_to_save, true_class=true_class)

        index += 1

    interpretability_results_file = "interpret_results_" + str(interpretability_step) + ".pickle"
    interpretability_to_save = base_directory + interpretability_directory + interpretability_model_directory + str(interpretability_step) + "/" + interpretability_results_file
    # pickle.dump(interpretability_results, open(interpretability_to_save, 'wb'))


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