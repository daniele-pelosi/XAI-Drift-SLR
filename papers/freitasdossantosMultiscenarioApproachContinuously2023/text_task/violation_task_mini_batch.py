import pathlib
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import util
import util_model

from transformers import AutoTokenizer, TFBertModel, TFRobertaModel, TFDistilBertModel, DistilBertConfig

import util_model_pt
from util import ResampleStrategy, LabelDescription


BERT_MODEL_NAME = "bert-base-uncased"
ROBERTA_MODEL_NAME = "roberta-base"
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
MAX_INPUT_LEN = 64
SPACY_MODEL_NAME = "en_core_web_trf"
CUDA_VISIBLE_DEVICES = ""

# To run TensorFlow on the CPU.
tf.config.set_visible_devices([], 'GPU')


def start_training(dataset_directory, dataset_file_name, number_k_datasets, save_experiment_directory, batch_size=32,
                   epochs=1, number_of_labels=1, data_block_size=1024, validation_done=True,
                   resample_strategy=ResampleStrategy.NoResample):
    base_directory = str(pathlib.Path(__file__).parent)
    complete_dataset_directory = base_directory + dataset_directory + dataset_file_name

    util.create_experiments_folders(save_experiment_directory, number_k_datasets)

    start_execution = 0
    for current_folder in range(start_execution, number_k_datasets):
        print("Execution: " + str(current_folder + 1) + " started!!!")

        current_dataset_path = complete_dataset_directory + str(current_folder) + ".pickle"
        current_dataset = pickle.load(open(current_dataset_path, "rb"))

        # Getting the separated datasets. Training and Testing were already specified with multi-label stratification.
        train_dataset = current_dataset["TRAIN_DATASET"]
        test_dataset = current_dataset["TEST_DATASET"]
        validation_dataset = current_dataset["VALIDATION_DATASET"]

        # Adding a column that indicates if the edit is a vandalism (only considering hate speech) or not.
        train_dataset = util.set_vandalism_label(train_dataset)
        test_dataset = util.set_vandalism_label(test_dataset)
        validation_dataset = util.set_vandalism_label(validation_dataset)

        labels_to_consider = [LabelDescription.REGULAR, LabelDescription.SWEARWORD,
                              LabelDescription.INSULT, LabelDescription.SEXUAL,
                              LabelDescription.RACISM, LabelDescription.HOMOPHOBIA,
                              LabelDescription.DERROGATORYTERMS, LabelDescription.SEXISM]

        # Getting the datasets only considering specific labels of interest (at this stage, the ones related to hate speech)
        train_dataset = util.consider_specific_labels(train_dataset, labels_to_consider)
        test_dataset = util.consider_specific_labels(test_dataset, labels_to_consider)
        validation_dataset = util.consider_specific_labels(validation_dataset, labels_to_consider)

        # Validation was done with another dataset, we can just incorporate for the final evaluation of the model.
        if validation_done:
            train_dataset = pd.concat([train_dataset, validation_dataset])

        train_vandalism = train_dataset[train_dataset["VANDALISM"] == 1]
        # Imbalance ratio 0.1.
        train_regular = train_dataset[train_dataset["VANDALISM"] == 0].sample(len(train_vandalism) * 10)  # 10 times the number of violations as the vandalism set.
        train_dataset = pd.concat([train_vandalism, train_regular]).sample(frac=1)

        # Since we apply a resampling strategy and mix two datasets randomly, it's important to save this dataset to use in other experiments for comparison's sake.
        # util.save_test(train_dataset, save_experiment_directory, current_folder, "undersampled_train_dataset.pickle")
        pickle.dump(test_dataset, open("interpretability_paper/test_violation.pickle", 'wb'))

        class_distribution = util.calculate_instances_binary(train_dataset, "VANDALISM")

        mini_batches = [train_dataset[index:index + data_block_size] for index in range(0, train_dataset.shape[0], data_block_size)]
        number_of_data_blocks = len(mini_batches)

        # We tokenize only the test set, the training set is going to be tokenized specifically for each data block.
        # tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)

        tokenized_test_data = util_model.tokenize_data(test_dataset, tokenizer, MAX_INPUT_LEN)
        y_test = np.array(test_dataset["VANDALISM"].values.tolist())

        # Pre-trained BERT model for tensorflow.
        # bert = TFBertModel.from_pretrained(BERT_MODEL_NAME)
        # roberta = TFRobertaModel.from_pretrained(ROBERTA_MODEL_NAME)

        distil_config = DistilBertConfig(dropout=0.1, attention_dropout=0.1, output_hidden_states=True)
        distilbert = TFDistilBertModel.from_pretrained(DISTILBERT_MODEL_NAME, config=distil_config)

        model = util_model.run_build_distil(distilbert, number_of_labels, "sigmoid")
        model, compile_time = util_model.run_compile(model)

        total_training_time = 0
        all_class_distribution_resample = []
        all_test_results = []
        for index_current_data_block, current_data_block in zip(range(0, number_of_data_blocks), mini_batches):
            current_data_block_size = len(current_data_block)
            if current_data_block_size > 0:
                data_block_class_distribution = util.calculate_instances_binary(current_data_block, "VANDALISM")

                # Executing the defined resample strategy.
                current_data_block = util.execute_resample_strategy(current_data_block, resample_strategy)
                data_block_class_distribution_resample = util.calculate_instances_binary(current_data_block, "VANDALISM")

                tokenized_data_block = util_model.tokenize_data(current_data_block, tokenizer, MAX_INPUT_LEN)
                y_train = np.array(current_data_block["VANDALISM"].values.tolist())

                print("Current data block: " + str(index_current_data_block))

                model, training_time = util_model.run_fit(model, tokenized_data_block, y_train, epochs, batch_size)

                # Get the results for each time data block the model is trained on. We want to see the evolution of
                # the results as new data blocks are used for training.
                evaluation_results, evaluation_predictions = util_model.run_evaluate(model, tokenized_test_data, y_test)
                individual_evaluation_result = util_model.get_individual_predictions_bin(test_dataset, evaluation_predictions, "VANDALISM")

                total_training_time += training_time

                data_block_test_results = {
                    "current_execution": current_folder,
                    "current_data_block": index_current_data_block,
                    "evaluation_results": evaluation_results,
                    "individual_evaluation_result": individual_evaluation_result,
                    "class_distribution": data_block_class_distribution,
                    "class_distribution_after_resample": data_block_class_distribution_resample,
                    "resample_strategy": resample_strategy.name,
                    "training_time": training_time,
                    "compile_time": compile_time
                }

                all_test_results.append(data_block_test_results)
                all_class_distribution_resample.append(data_block_class_distribution_resample)

                file_name = "test_results_" + str(index_current_data_block) + ".pickle"
                util.save_test(data_block_test_results, save_experiment_directory, current_folder, file_name)

        class_distribution_resample = np.sum(all_class_distribution_resample, axis=0)

        # The last test_result in all_test_results contains the test when the model is trained on all data points.
        overall_test_results = {
            "current_execution": current_folder,
            "evaluation_results": all_test_results[number_of_data_blocks - 1]["evaluation_results"],
            "individual_evaluation_result": all_test_results[number_of_data_blocks - 1]["individual_evaluation_result"],
            "class_distribution": class_distribution,
            "class_distribution_after_resample": class_distribution_resample,
            "training_time": total_training_time,
            "compile_time": compile_time,
            "number_of_data_blocks": number_of_data_blocks
        }

        print("Training Time: " + str(training_time))

        util.save_test(overall_test_results, save_experiment_directory, current_folder)
        util.save_model(model, save_experiment_directory, current_folder)


def start_training_interpretability(dataset_directory, dataset_file_name, save_experiment_directory, model_name, batch_size=32,
                                    epochs=1, number_of_labels=2, data_block_size=1024, resample_strategy=ResampleStrategy.NoResample,
                                    load_train_dataset=False):
    base_directory = str(pathlib.Path(__file__).parent)
    complete_dataset_directory = base_directory + dataset_directory + dataset_file_name

    print("Interpretability Execution Started!!!")

    if not load_train_dataset:
        current_dataset_path = complete_dataset_directory + str(0) + ".pickle"
        current_dataset = pickle.load(open(current_dataset_path, "rb"))

        # Getting the separated datasets. Training and Testing were already specified with multi-label stratification.
        train_dataset = current_dataset["TRAIN_DATASET"]
        test_dataset = current_dataset["TEST_DATASET"]
        validation_dataset = current_dataset["VALIDATION_DATASET"]

        # Adding a column that indicates if the edit is a vandalism (only considering hate speech) or not.
        train_dataset = util.set_vandalism_label(train_dataset)
        test_dataset = util.set_vandalism_label(test_dataset)
        validation_dataset = util.set_vandalism_label(validation_dataset)

        labels_to_consider = [LabelDescription.REGULAR, LabelDescription.SWEARWORD,
                              LabelDescription.INSULT, LabelDescription.SEXUAL,
                              LabelDescription.RACISM, LabelDescription.HOMOPHOBIA,
                              LabelDescription.DERROGATORYTERMS, LabelDescription.SEXISM]

        # Getting the datasets only considering specific labels of interest (at this stage, the ones related to hate speech)
        train_dataset = util.consider_specific_labels(train_dataset, labels_to_consider)
        test_dataset = util.consider_specific_labels(test_dataset, labels_to_consider)
        validation_dataset = util.consider_specific_labels(validation_dataset, labels_to_consider)

        train_dataset = pd.concat([train_dataset, validation_dataset, test_dataset])

        train_vandalism = train_dataset[train_dataset["VANDALISM"] == 1]
        train_regular = train_dataset[train_dataset["VANDALISM"] == 0].sample(len(train_vandalism) * 10)  # Imbalance ratio 0.1.
        train_dataset = pd.concat([train_vandalism, train_regular]).sample(frac=1)

        # Since we apply a resampling strategy and mix two datasets randomly, then it's important to save this dataset to use in other experiments for comparison's sake.
        # util.save_test(train_dataset, save_experiment_directory, current_folder, "undersampled_train_dataset.pickle")
        pickle.dump(train_vandalism, open("interpretability_paper_violation/train_only_violation.pickle", 'wb'))
        pickle.dump(train_dataset, open("interpretability_paper_violation/train_complete_violation.pickle", 'wb'))
    else:
        train_dataset = pickle.load(open("interpretability_paper_violation/train_complete_violation.pickle", 'rb'))

    class_distribution = util.calculate_instances_binary(train_dataset, "VANDALISM")

    mini_batches = [train_dataset[index:index + data_block_size] for index in range(0, train_dataset.shape[0], data_block_size)]
    number_of_data_blocks = len(mini_batches)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = util_model_pt.define_model_class(model_name, number_of_labels, problem_type="single_label_classification")

    total_training_time = 0
    all_class_distribution_resample = []
    all_test_results = []
    for index_current_data_block, current_data_block in zip(range(0, number_of_data_blocks), mini_batches):
        current_data_block_size = len(current_data_block)
        if current_data_block_size > 0:
            data_block_class_distribution = util.calculate_instances_binary(current_data_block, "VANDALISM")

            # Executing the defined resample strategy.
            current_data_block = util.execute_resample_strategy(current_data_block, resample_strategy)
            data_block_class_distribution_resample = util.calculate_instances_binary(current_data_block, "VANDALISM")

            tokenized_data_block = util_model_pt.convert_to_pt_dataset(current_data_block, "VANDALISM", tokenizer, "single")
            # y_train = np.array(current_data_block["labels"].values.tolist())

            print("Current data block: " + str(index_current_data_block))

            label_names = ["0", "1"]
            model, trainer = util_model_pt.run_fit(model, save_experiment_directory, tokenized_data_block, tokenizer, epochs, batch_size,
                                                   label_names)

    util_model_pt.save_model_interpretability(model, save_experiment_directory)


