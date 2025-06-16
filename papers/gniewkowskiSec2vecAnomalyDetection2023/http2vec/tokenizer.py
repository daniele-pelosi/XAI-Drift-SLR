import argparse
import json
import os

from tokenizers import CharBPETokenizer
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast
from http2vec.utils import file_generator

class Tokenizer(object):

    def __init__(self, name, only_train=False, **args):

        self.name = name
        tokenizers = {
            "char_bpe": CharBPETokenizer,
            "byte_bpe": ByteLevelBPETokenizer,
        }
        self.only_train = only_train
        self.tokenizer = tokenizers[name](**args)

    def train(self, files, **args):

        def line_generator(files):
            for fn in file_generator(files):
                if self.only_train and "test_" in fn:
                    continue
                with open(fn) as f:
                    data = f.readlines()
                data = [json.loads(d)["text"] for d in data]
            for doc in data:
                yield doc

        self.tokenizer.train_from_iterator(
            iterator=line_generator(files),
            vocab_size=52_000,
            min_frequency=2,
            special_tokens=[
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "<mask>",
            ]
        )

    def dump(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.tokenizer.save_model(output_dir)


def get_args():

    parser = argparse.ArgumentParser(description="Tokenizer")

    parser.add_argument(
        "input_data", help="Text to tokenize"
    )

    parser.add_argument(
        "tokenizer", help="Path to trained tokenizer",
    )

    return parser.parse_args()


def main():
    '''Prints tokenized text'''
    args = get_args()

    tokenizer = RobertaTokenizerFast.from_pretrained(
        args.tokenizer,
        max_len=512
    )

    with open(args.input_data, "rt") as fp:
        lines = fp.readlines()

    for l in lines:
        print(tokenizer.tokenize(l))


if __name__ == "__main__":
    main()
