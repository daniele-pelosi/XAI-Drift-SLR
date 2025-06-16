import argparse
import os

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
from http2vec.dataset import CustomLineByLineTextDataset
from http2vec.models import (
    Roberta,
    BoW,
    FastText,
    Elmo
)


def get_args():
    parser = argparse.ArgumentParser(description="Language model for HTTP")
    parser.add_argument(
        "input_data", help="Data used for training"
    )
    parser.add_argument(
        "output_path", help="Output for train model"
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
        "--tokenizer", help="Tokenizer"
    )
    parsed, unknow = parser.parse_known_args()

    def convert_argument(arg):
        try:
            int(arg)
        except ValueError:
            pass
        else:
            return int(arg)

        try:
            float(arg)
        except ValueError:
            pass
        else:
            return float(arg)

        return arg

    extra = dict()
    for key, value in zip(*[iter(unknow)]*2):
        key = key.lstrip("-")
        extra[key] = convert_argument(value)

    return parsed, extra


def main():
    models = {
        "roberta": Roberta,
        "bow": BoW,
        "fasttext": FastText,
        "elmo": Elmo,
    }
    args, extra = get_args()

    tokenizer = RobertaTokenizerFast.from_pretrained(
        args.tokenizer,
        max_len=512
    )
    dataset = CustomLineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=args.input_data,
        block_size=128,
    )

    model = models[args.model](**extra)
    model.train(dataset)
    model.save(args.output_path)


if __name__ == "__main__":
    main()
