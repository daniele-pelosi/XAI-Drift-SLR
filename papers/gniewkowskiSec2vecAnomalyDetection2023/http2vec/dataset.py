import argparse
import os
import json

import torch
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import PreTrainedTokenizerFast
from transformers import RobertaModel
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments

from torch.utils.data.dataset import Dataset
from http2vec.utils import file_generator


class CustomLineByLineTextDataset(Dataset):

    def __init__(self, tokenizer, file_path: str, block_size: int):

        lines = []
        for filename in file_generator(file_path):
            print(filename)
            with open(filename, encoding="utf-8") as f:
                data = f.read().splitlines()
            data = [json.loads(d)["text"] for d in data]
            data = [d.splitlines(True) for d in data]
            data = [item for sublist in data for item in sublist]
            lines += [line for line in data if (len(line) > 0 and not line.isspace())]

        self.tokenizer = tokenizer
        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

    def get_line(self, i, as_ids=False):
        line = self.examples[i]["input_ids"].tolist()
        return self.tokenizer.convert_ids_to_tokens(line)

    def get_tokenizer(self):
        return self.tokenizer
