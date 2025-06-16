'''
CharBPETokenizer: The original BPE
ByteLevelBPETokenizer: The byte level version of the BPE
SentencePieceBPETokenizer: A BPE implementation compatible with the one used by SentencePiece
BertWordPieceTokenizer: The famous Bert tokenizer, using WordPiece
'''
import argparse
import os

from http2vec.utils import load_config
from http2vec.tokenizer import Tokenizer


def get_args():

    parser = argparse.ArgumentParser(description="Tokenizer")

    parser.add_argument(
        "input_data", help="Data used for training"
    )
    parser.add_argument(
        "output_path", help="Path used for saving trained tokenizer"
    )
    parser.add_argument(
        "--method", help="Name of Tokenizer",
        choices=[
           "char_bpe", "byte_bpe",
        ]
    )
    parser.add_argument(
        "--only-train", help="If use test data for training.",
        action="store_true"
    )


    return parser.parse_args()


def main():
    args = get_args()

    tokenizer = Tokenizer(
        args.method,
        only_train=args.only_train
    )
    tokenizer.train(
        files=args.input_data,
    )
    tokenizer.dump(args.output_path)


if __name__ == "__main__":
    main()
