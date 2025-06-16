import functools
import pathlib

import torch

from sklearn.metrics import classification_report
from torch.optim import AdamW
from transformers import RobertaConfig, RobertaForSequenceClassification, DistilBertConfig, DistilBertForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, \
    TextClassificationPipeline
from datasets import load_dataset, Dataset

ROBERTA_MODEL_NAME = "roberta-base"
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"


def define_model_class(model_name, number_of_labels, problem_type="multi_label_classification", id2label={0: 0, 1: 1},
                       label2id={0: 0, 1: 1}):
    # Device configuration. Use GPU or CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == ROBERTA_MODEL_NAME:
        roberta_config = RobertaConfig.from_pretrained(ROBERTA_MODEL_NAME, num_labels=number_of_labels,
                                                       problem_type=problem_type, id2label=id2label)
        return RobertaForSequenceClassification.from_pretrained(ROBERTA_MODEL_NAME, config=roberta_config).to(device)
    elif model_name == DISTILBERT_MODEL_NAME:
        distil_config = DistilBertConfig(dropout=0.1, attention_dropout=0.1, output_hidden_states=True, num_labels=number_of_labels,
                                         problem_type=problem_type, id2label=id2label, label2id=label2id)
        return DistilBertForSequenceClassification.from_pretrained(model_name, config=distil_config).to(device)


def define_tokenizer(model_name, max_input_len=64):
    return AutoTokenizer.from_pretrained(model_name, add_special_tokens=True,
                                         max_length=max_input_len,
                                         truncation=True,
                                         padding="max_length",
                                         return_tensors="pt",
                                         return_token_type_ids=False,
                                         return_attention_mask=True,
                                         verbose=True)


def tokenize_data(dataset, tokenizer, text_label="NEW_FORMATTED_TEXT", max_input_len=64):
    # Tokenizing the text data
    tokenized_data = tokenizer(dataset[text_label],
                               add_special_tokens=True,
                               max_length=max_input_len,
                               truncation=True,
                               padding="max_length",
                               return_tensors="pt",
                               return_token_type_ids=False,
                               return_attention_mask=True,
                               verbose=True)

    return tokenized_data


def run_compile(model):
    optimizer = AdamW(model.parameters(),
                      lr=1e-4,  # 5e-5
                      eps=1e-08,
                      weight_decay=0.01)

    return optimizer


def convert_to_pt_dataset(dataset, label_column, tokenizer, problem_type="multilabel"):
    # Changing to Float because the trainer() method requires a TensorFloat for the multilabel case.
    if problem_type == "multilabel":
        dataset[label_column] = dataset[label_column].apply(lambda labels: [float(label_val) for label_val in labels])

    dataset.rename({label_column: 'labels'}, axis=1, inplace=True)
    converted_dataset = Dataset.from_pandas(dataset)

    tokenized_data = converted_dataset.map(functools.partial(tokenize_data, tokenizer=tokenizer), batched=True)
    tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized_data


def run_fit(model, save_directory, train_dataset, tokenizer, num_epochs=3, batch_size=32, label_names=["0", "1", "2", "3", "4", "5"]):
    training_args = TrainingArguments(output_dir=save_directory, learning_rate=1e-4, weight_decay=0.01, per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size, num_train_epochs=num_epochs, evaluation_strategy="no",
                                      remove_unused_columns=False, overwrite_output_dir=True, label_names=label_names)

    # The default optimizer is AdamW with the scheduler given by get_linear_schedule_with_warmup().
    # We pass the tokenizer because padding might be applied in this step (giving the same size to every sample).
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    trainer.train()

    return model, trainer


def run_evaluate(model, tokenizer, test_data, y_test, max_input_len=64):
    tokenizer_parameters = {"add_special_tokens": True,
                            "max_length": max_input_len,
                            "truncation": True,
                            "padding": "max_length",
                            "return_token_type_ids": False,
                            "return_attention_mask": True,
                            "verbose": True}

    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=model.device, **tokenizer_parameters)
    classifier_output = classifier(test_data)
    y_predicted = [output['label'] for output in classifier_output]

    calculated_classification_report = classification_report(y_test, y_predicted, output_dict=True, zero_division=0)
    return calculated_classification_report, y_predicted


def save_model_interpretability(trained_model, save_experiment_directory, trained_model_folder_name="trained_model/"):
    base_directory = str(pathlib.Path(__file__).parent)
    file_path = save_experiment_directory + "/" + trained_model_folder_name + "interpretability_model.pt"
    complete_path_to_save = base_directory + file_path

    torch.save(trained_model.state_dict(), complete_path_to_save)


def load_model_interpretability(new_model, save_experiment_directory, trained_model_folder_name="trained_model/"):
    base_directory = str(pathlib.Path(__file__).parent)
    file_path = save_experiment_directory + trained_model_folder_name + "interpretability_model.pt"
    complete_path_to_load = base_directory + file_path

    new_model.load_state_dict(torch.load(complete_path_to_load))
    return new_model

