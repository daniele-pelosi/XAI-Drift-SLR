from collections import defaultdict
import pickle
import os

import json
import numpy as np
from transformers import RobertaTokenizerFast

from http2vec.models.base import BaseModel
from http2vec.dataset import CustomLineByLineTextDataset


class BoW(BaseModel):

    def __init__(self, length=3072):
        super().__init__()
        self.model = None
        self.length = length

    def train(self, dataset):

        sentences = [dataset.get_line(i) for i in range(len(dataset))]
        top_n_words = defaultdict(int)
        for sentence in sentences:
            for word in sentence:
                top_n_words[word] += 1

        top_n_words = sorted(
            top_n_words.items(),
            key=lambda t: t[1],
            reverse=True
        )[:self.length-1]
        self.model = [w[0] for w in top_n_words]

    def save(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        with open(f"{output_path}/bow.pickle", "wb") as f:
            pickle.dump(self.model, f)

    def load(self, model_path):
        with open(f"{model_path}/bow.pickle", "rb") as f:
            self.model = pickle.load(f)

    def vectorize(self, document, tokenizer):
        """Vectorize a document.
        
        document: list of list of words.
        
        Returns a vector (the last position is number of oov).
        """
        oov_words = 0
        vector = np.zeros(len(self.model) + 1)
        for sentence in document:
            sentence = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )["input_ids"].tolist()[0]
            sentence = tokenizer.convert_ids_to_tokens(sentence)
            for word in sentence:
                if word in self.model:
                    vector[self.model.index(word)] += 1
                else:
                    vector[-1] += 1
        return vector


    def vectorize2(self, document, tokenizer):
        """Vectorize a document.
        
        document: list of list of words.
        
        Returns a vector (the last position is number of oov).
        """
        oov_words = 0
        vector = np.zeros(len(self.model) + 1)
        for sentence in document:
            sentence = tokenizer.convert_ids_to_tokens(sentence)
            for word in sentence:
                if word in self.model:
                    vector[self.model.index(word)] += 1
                else:
                    vector[-1] += 1
        return vector


if __name__ == "__main__":
    model = BoW()
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "data/tokenizers/CSIC2010",
        max_len=512
    )
    dataset = CustomLineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="data/datasets/CSIC2010/train_model.jsonl",
        block_size=128,
    )
    model.train(dataset)
    model.save("tmp/bow")
    model = BoW()
    model.load("tmp/bow")
    with open("data/datasets/CSIC2010/test_model.jsonl") as f:
        document = json.loads(f.readlines()[0])["text"]
    print(model.vectorize(document, tokenizer))

