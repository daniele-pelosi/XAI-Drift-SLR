# In this project we create the models used to classify one edit as violation or regular, besides also getting the
# tag for the cases in which the edit is a vandalism.

import interpretability_binary
import global_interpretability
import interpretability_multilabel
import multilabel_task_mini_batch
import violation_task_mini_batch

BERT_MODEL_NAME = "bert-base-uncased"
ROBERTA_MODEL_NAME = "roberta-base"
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"

if __name__ == '__main__':
    # region Experiment
    dataset_directory = "/experiments/datasets/experiment_1/"
    dataset_file_name = "folder_"
    number_k_datasets = 10
    # endregion

    # Binary case with mini-batch, RoBERTa.
    '''violation_task_mini_batch.start_training_interpretability(dataset_directory, dataset_file_name,
                                                              "/experiments/violation_task_mini_batch/interpretability_run/",
                                                              ROBERTA_MODEL_NAME)'''

    # Binary case with mini-batch, DistilBERT.
    '''violation_task_mini_batch.start_training_interpretability(dataset_directory, dataset_file_name,
                                                              "/experiments/violation_task_mini_batch_distil/interpretability_run/",
                                                              DISTILBERT_MODEL_NAME, load_train_dataset=True)'''

    # Multi-label with mini-batch, RoBERTa
    # multilabel_task_mini_batch.start_training(dataset_directory, dataset_file_name, number_k_datasets,
    #                                          "/experiments/multilabel_task_mini_batch/k_run/")

    '''multilabel_task_mini_batch.start_training_interpretability(dataset_directory, dataset_file_name,
                                                               "/experiments/multilabel_task_mini_batch/interpretability_run/",
                                                               ROBERTA_MODEL_NAME, load_train_dataset=True)'''

    # Multi-label with mini-batch, DistilBERT
    # multilabel_task_mini_batch.start_training(dataset_directory, dataset_file_name, number_k_datasets,
    #                                          "/experiments/multilabel_task_mini_batch_distil/k_run/")

    '''multilabel_task_mini_batch.start_training_interpretability(dataset_directory, dataset_file_name,
                                                               "/experiments/multilabel_task_mini_batch_distil/interpretability_run/",
                                                               DISTILBERT_MODEL_NAME, load_train_dataset=True)'''

    # Interpretability
    # interpretability_binary.execute_interpretability_from_disk_all(dataset_directory, "/experiments/violation_task_mini_batch_distil/k_run/",
    #                                                                dataset_file_name, 0, "interpretability_paper/", 0)
    # interpretability_binary.execute_interpretability_from_disk_all(dataset_directory, "/experiments/violation_task_mini_batch/k_run/",
    #                                                                dataset_file_name, 0, "interpretability_paper/", 0)

    # Generating the local interpretability for the binary case.
    '''interpretability_binary.execute_interpretability_complete_data(dataset_directory, "/experiments/violation_task_mini_batch/interpretability_run/",
                                                                   "train_only_violation.pickle", "/interpretability_paper_violation/", 0,
                                                                   "roberta/", ROBERTA_MODEL_NAME, 2)'''

    '''interpretability_binary.execute_interpretability_complete_data(dataset_directory, "/experiments/violation_task_mini_batch_distil/interpretability_run/",
                                                                   "train_only_violation.pickle", "/interpretability_paper_violation/", 0,
                                                                   "distil_bert/", DISTILBERT_MODEL_NAME, 2)'''

    # Generating the local interpretability for the multi-label case.
    '''interpretability_multilabel.execute_interpretability_complete_data("/experiments/multilabel_task_mini_batch/interpretability_run/",
                                                                       "train_complete_multilabel.pickle", "/interpretability_paper_violation/multi/", 0,
                                                                       "roberta/", ROBERTA_MODEL_NAME, 6)'''

    '''interpretability_multilabel.execute_interpretability_complete_data("/experiments/multilabel_task_mini_batch_distil/interpretability_run/",
                                                                       "train_complete_multilabel.pickle", "/interpretability_paper_violation/multi/", 0,
                                                                       "distil/", DISTILBERT_MODEL_NAME, 6)'''

    # Generating the sum of relevance score graphs for the binary case.
    # global_interpretability.generate_global_chart("/interpretability_paper_violation/roberta/0/", 0, "RoBERTa")
    # global_interpretability.generate_global_chart("/interpretability_paper_violation/distil_bert/0/", 0, "DistilBERT")

    # Generating the sum of relevance score graphs for the multi-label case.
    '''labels_description = ["Swear", "Insult", "Sexual", "Racism", "LGBTIQA+", "Misogyny"]
    global_interpretability.generate_global_interpretability_multi("/interpretability_paper_violation/multi/roberta/0/", 0, "RoBERTa", 6, labels_description)
    global_interpretability.generate_global_interpretability_multi("/interpretability_paper_violation/multi/distil/0/", 0, "DistilBERT", 6, labels_description)'''
