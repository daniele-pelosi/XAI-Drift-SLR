import argparse
from transformers import RobertaTokenizer, RobertaModel
from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
import sys
import numpy as np
import scipy.spatial


def get_args():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "input_data", help=""
    )
    parser.add_argument(
        "output_path", help=""
    )
    parser.add_argument(
        "--context", help="The vector space from DVC stage 3"
    )
    parser.add_argument(
        "--tokenizer", help="Path to pretrained tokenizer"
    )
    parser.add_argument(
        "--model", help="Path to pretrained RoBERTa model"
    )

    return parser.parse_args()


def read_file(filename):
    with open(filename) as f:
        data = f.readlines()
    return data


def vectorize(doc, model_path, tokenizer_path):

    device = "cpu"

    tokenizer = RobertaTokenizer.from_pretrained(
        tokenizer_path
    )
    model = RobertaModel.from_pretrained(
        model_path,
        output_hidden_states=True
    ).to(device)

    doc_vectors = []
    for line in doc:
        inputs = tokenizer(
            line,
            return_tensors="pt",
            max_length=512,
        ).to(device)
        outputs = model(**inputs)
        last_four_layers = outputs.hidden_states[-4:]
        cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
        cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze()
        doc_vectors.append(cat_sentence_embedding)

    doc_vectors = torch.stack(doc_vectors)
    doc_embeding = torch.mean(doc_vectors, dim=0).cpu().detach().numpy()
    return doc_embeding

    docs = pd.DataFrame(np.stack(docs))


def get_closest(vector, context, n_neighbors=500):

    neigh = NearestNeighbors()
    neigh.fit(context)

    closest_n = neigh.kneighbors(
        vector.reshape(1, -1),
        n_neighbors=n_neighbors,
        return_distance=False
    ).tolist()
    return closest_n[0]


def main():

    args = get_args()
    with open(args.context) as f:
        data = json.load(f)

    labels = data["labels"]
    texts = data["texts"]
    data = data["data"]
    data = pd.DataFrame.from_dict(data)
    
    filename = args.input_data
    text = read_file(filename)
    vector = vectorize(text, args.model, args.tokenizer)
    closest = get_closest(vector, data.to_numpy())
    
    print("CLOSEST SAMPLES")
    for i, c in enumerate(closest):
        print(f"   {i}: {texts[c]}")
    

if __name__ == "__main__":
    main()
