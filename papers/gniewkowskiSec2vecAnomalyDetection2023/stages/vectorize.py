import argparse
from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os

from http2vec.utils import file_generator
from http2vec.models import (
    Roberta,
    BoW,
    FastText,
    Elmo
)


def get_args():

    parser = argparse.ArgumentParser(description="Tokenizer")

    parser.add_argument(
        "input_data", help="Data used for vectorization"
    )
    parser.add_argument(
        "output_path", help="Output path for vectors"
    )
    parser.add_argument(
        "--model", help="Model name",
        choices=[
           "roberta",
           "bow",
           "fasttext",
           "elmo",
        ]
    )
    parser.add_argument(
        "--model-path", help="Path to a model",
    )
    parser.add_argument(
        "--tokenizer", help="Path to a tokenizer"
    )
    return parser.parse_args()


def read_file(filename):
    with open(filename) as f:
        data = f.readlines()

    data = [json.loads(d) for d in data]
    return data


def main():

    args = get_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    models = {
        "roberta": Roberta,
        "bow": BoW,
        "fasttext": FastText,
        "elmo": Elmo,
    }


    tokenizer = RobertaTokenizer.from_pretrained(
        args.tokenizer,
    )
    model = models[args.model]()
    model.load(args.model_path)
    dataset = read_file(f"{args.input_data}")

    labels = []
    docs = []
    texts = []

    for document in tqdm(list(dataset)):
        texts.append(document["id"])
        labels.append(document["label"])
        docs.append(
            model.vectorize(document["text"].splitlines(True), tokenizer)
        )

    docs = pd.DataFrame(np.stack(docs))
    result = {
        "labels": labels,
        "data": docs.to_dict(),
        "ids": texts
    }

    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "result.json"), "wt") as fp:
        json.dump(result, fp)

if __name__ == "__main__":
    main()
