import json
import os
import pickle

import pandas as pd
import numpy as np


from http2vec.models import (
    Roberta,
    BoW,
    FastText
)
from transformers import RobertaTokenizer


def get_vectorizer(language_model, dataset, size):
    """Returns a vectorizer (trained language model)"""
    model_path = f"data/models/{language_model}-{dataset}-{size}"
    models = {
        "roberta": Roberta,
        "bow": BoW,
        "fasttext": FastText,
    }
    model = models[language_model]()
    model.load(model_path)
    return model


def get_classifier_fn(vectorizer, tokenizer_path, classifier, mode="rp"):
    """Returns a function that takes list of docs and classify them.
    
    mode: rp (return proba), df (decision_function), predict
    
    """
    tokenizer = RobertaTokenizer.from_pretrained(
        tokenizer_path,
    )
    
    def fun(docs):
        result = []
        for doc in docs:
            doc = doc.splitlines()
            vector = vectorizer.vectorize(doc, tokenizer)
            if classifier.classes_[0] == "anomaly":
                flip = True


            if mode == "rp":
                pred = classifier.predict_proba(vector.reshape(1, -1))[0]
                pred = np.flip(pred) if flip else pred
                result.append(pred)
            elif mode == "df":
                pred = classifier.decision_function(vector.reshape(1, -1))[0]
                pred = -pred if flip else pred
                result.append(pred)
            else:
                pred = classifier.predict(vector.reshape(1, -1))[0]
                result.append(pred)
        return np.array(result)
    return fun



def metrics_generator(language_model, dataset, classifier="lr"):
    """Generate DataFrame with metrics.
    Args:
        language_model (str): roberta, bow
        dataset (str): name of dateset like bow-CSIC2010
    
    Returns: DataFrame
    """
    ms = []
    lens = []

    dataset = f"{language_model}-{dataset}"
    for filename in os.listdir("data/classification/"):
        length = filename.split("-")[-1]
        if filename.startswith(dataset) and length.isdigit():
            length = int(length)
            try:
                with open(f"data/classification/{filename}/metrics.json") as f:
                    metrics = json.load(f)[classifier]
            except:
                continue
            ms.append(metrics)
            lens.append(length)
            
    ms = pd.DataFrame(ms)
    ms["length"] = lens
    ms = ms.sort_values(by="length")
    ms.style.set_caption(dataset)
    return ms


def get_classifier(
    language_model,
    dataset,
    size,
    clf_name="lr"
):
    """Returns pretrained classifier.
    
    Args
        language_model (str): bow, roberta, etc.
        dataset (str): name of the dataset (CSIC2010),
        size (int): size of the used language model
    """
    assert clf_name in ["lr", "svc", "rf"]

    exp = f"{language_model}-{dataset}-{size}"
    pretrained_classifier = f"data/classification/{exp}/saved/{clf_name}.pickle"
    with open(pretrained_classifier, "rb") as f:
        clf = pickle.load(f)
    return clf


def get_vectors(
    language_model,
    dataset,
    size,
    return_all=True
):
    """Gets vectors.
    
    Args
        language_model (str): bow, roberta, etc.
        dataset (str): name of the dataset (CSIC2010),
        size (int): size of the used language model
        return_all (bool): 
            if true return all vectors
            if false return only those used in clf testing
    Returns:
        data (numpy): vectors
        labels (numpy): labels
        ids (numpy): ids of samples
    """


    exp = f"{language_model}-{dataset}-{size}"
        
    split = f"data/classification/{exp}/saved/split.json"
    with open(split) as f:
        data = json.load(f)
    test_ids = data["test"]

    vector_path =  f"data/vectors/{exp}/result.json"
    with open(vector_path) as f:
        data = json.load(f)
    labels = np.array(data["labels"])
    ids = np.array(data["ids"], dtype=str)
    data = pd.DataFrame.from_dict(data["data"]).to_numpy()

    if not return_all:
        test_index = np.isin(ids, test_ids)
        data = data[test_ids, :]
        labels = labels[test_ids]
        ids = ids[test_ids]

    return data, labels, ids


def get_data(dataset):
    """Loads part of the dataset that was used for classification.
    
    Returns (data: list, labels: list)
    """
    data_path = f"data/datasets/{dataset}/test_model.jsonl"
    df = pd.read_json(data_path, orient='records', lines=True)
    
    if "attack_cat" not in df:
        df["attack_cat"] = ""

    # New
    df.index = df["id"].astype(str)
    return df


def get_split(dataset, language_model, size):
    exp = f"bow-{dataset}-3072"
    data_path = f"data/classification/{exp}/saved/split.json"
    with open(data_path) as f:
        data = json.load(f)
    return data["train"], data["test"]


def get_predictions(dataset, language_model, size, classifier="lr"):
    exp = f"{language_model}-{dataset}-{size}"
    data_path = f"data/classification/{exp}/saved/predictions-{classifier}.json"
    with open(data_path) as f:
        data = json.load(f)
    return data
