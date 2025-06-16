import argparse
from gc import callbacks
import os
import string
import random

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
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data.dataset import Dataset
from http2vec.utils import file_generator
from http2vec.models.base import BaseModel
from http2vec.dataset import CustomLineByLineTextDataset
import json


class Roberta(BaseModel):

    def __init__(self, **kwargs):
        super().__init__()
        self.model = None
        self.kwargs = self._get_defaults(**kwargs)

    def _get_defaults(self, **kwargs):

        if "length" in kwargs:
            length = kwargs["length"]
            kwargs["hidden_size"] = int(length/4)
            del kwargs["length"]

        defaults = {
            "vocab_size": 52000,
            "max_position_embeddings": 514,
            "hidden_size": 768, # Final vector is 4 times this value
            "num_attention_heads": 12,
            "num_hidden_layers": 6,
            "type_vocab_size": 1,
        }
        result = dict()
        for k, v in defaults.items():
            if k not in kwargs:
                result[k] = v
            else:
                result[k] = kwargs[k]
        return result

    def train(self, dataset):
        self.tokenizer = dataset.get_tokenizer()

        roberta_config = RobertaConfig(
            **self.kwargs
        )
        self.model = RobertaForMaskedLM(config=roberta_config)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )
        r_str = ''.join(random.choice(string.ascii_uppercase + string.digits)
                for _ in range(5))
        training_args = TrainingArguments(
            output_dir="./tmp/roberta" + r_str,
            overwrite_output_dir=True,
            num_train_epochs=20,
            per_gpu_train_batch_size=32,
            save_steps=1000,
            save_total_limit=1,
            prediction_loss_only=True,
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
            callbacks=[TensorBoardCallback(tb_writer=SummaryWriter())]
        )
        self.trainer.train()

    def save(self, output_path):
        self.trainer.save_model(output_path)

    def load(self, model_path):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        #device = "cpu"
        self.model = RobertaModel.from_pretrained(
            model_path,
            output_hidden_states=True
        ).to(device)

    def vectorize(self, document, tokenizer):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        #device = "cpu"
        doc_vectors = []
        for line in document:
            inputs = tokenizer(
                line,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            outputs = self.model(**inputs)
            last_four_layers = outputs.hidden_states[-4:]
            cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
            cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze()
            doc_vectors.append(cat_sentence_embedding)

        doc_vectors = torch.stack(doc_vectors)
        doc_embeding = torch.mean(doc_vectors, dim=0)
        return doc_embeding.cpu().detach().numpy()



if __name__ == "__main__":
    model = Roberta()
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
    model.save("tmp/roberta")
    model = Roberta()
    model.load("tmp/roberta")
    with open("data/datasets/CSIC2010/test_model.jsonl") as f:
        document = json.loads(f.readlines()[0])['text']
    print(model.vectorize(document, tokenizer))
