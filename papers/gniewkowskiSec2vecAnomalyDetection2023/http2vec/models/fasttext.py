import os
import json

import numpy as np
from transformers import RobertaTokenizerFast
from gensim import models

from http2vec.models.base import BaseModel
from http2vec.dataset import CustomLineByLineTextDataset


class FastText(BaseModel):

    def __init__(self, length=3072):
        super().__init__()
        self.model = None
        self.length = length

    def train(self, dataset):
        self.model = models.FastText(vector_size=self.length, min_n=1, max_n=6)
        sentences = [dataset.get_line(i) for i in range(len(dataset))]

        self.model.build_vocab(sentences)
        self.model.train(
            corpus_iterable=sentences,
            total_examples=len(sentences),
            epochs=20
        )

    def save(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        self.model.save(f"{output_path}/fasttext.model")

    def load(self, model_path):
        self.model = models.FastText.load(f"{model_path}/fasttext.model")

    def vectorize(self, document, tokenizer):
        """Vectorize a document.
        
        document: list of strings.
        
        Returns a vector (the last position is number of oov.
        """
        sentence_vectors = list()
        for sentence in document:
            sentence = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )["input_ids"].tolist()[0]
            sentence = tokenizer.convert_ids_to_tokens(sentence)
            word_vectors = list()
            for word in sentence:
                word_vectors.append(self.model.wv[word])
            sentence_vectors.append(np.mean(word_vectors, axis=0))

        return np.mean(sentence_vectors, axis=0)



if __name__ == "__main__":
    model = FastText()
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
    model.save("tmp/fasttext")
    model = FastText()
    model.load("tmp/fasttext")
    with open("data/datasets/CSIC2010/test_model.jsonl") as f:
        document = json.loads(f.readlines()[0])['text']
    print(model.vectorize(document, tokenizer))

