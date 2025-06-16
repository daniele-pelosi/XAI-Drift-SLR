import numpy as np
import os
import torch

from allennlp.data import Batch
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.models.archival import archive_model, load_archive
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder
from allennlp.modules.seq2vec_encoders import CnnHighwayEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, EmptyEmbedder, TokenCharactersEncoder
from allennlp.nn.util import move_to_device
from allennlp_models.lm.models import LanguageModel
from allennlp_models.lm.modules.seq2seq_encoders import\
    BidirectionalLanguageModelTransformer
from allennlp.training.trainer import GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from http2vec.dataset import CustomLineByLineTextDataset
from http2vec.models.base import BaseModel
from transformers import RobertaTokenizerFast

from typing import List


class Elmo(BaseModel):

    def __init__(self, dataset=None, tokenizer=None, **kwargs):
        super().__init__()

        if len(kwargs) == 0:
            kwargs = {
                "embedding_dim": 16,
                "filters": [
                        [1, 32],
                        [2, 32],
                        [3, 64],
                        [4, 128],
                        [5, 256],
                        [6, 512],
                        [7, 1024]
                    ],
                "enc_num_highway": 2,
                "enc_projection_dim": 512,
                "enc_activation": "relu",
                "enc_projection_location": "after_highway",
                "enc_do_layer_norm": True,
                "con_type": 'LSTM',
                "con_input_size": 512,
                "con_hidden_size": 2048,
                "con_num_layers": 2,
                "con_bias": True,
                "con_dropout": 0.0,
                "con_bidirectional": True,
                "con_stateful": False,
                "con_input_dropout": None,
                "con_return_all_layers": False,
                "mod_dropout": 0.1,
                "mod_num_samples": 8192,
                "mod_sparse_embeddings": False,
                "mod_bidirectional": True
            }
        
        # Vocab copy and generation.
        self._vocab = Vocabulary.empty()
        _characters = {"<s>", "<pad>", "</s>", "<unk>", "<mask>"}
        self._vocab.add_tokens_to_namespace(_characters, "tokens")
        if dataset is not None:
            _tokenizer_vocab = dataset.tokenizer.get_vocab()
            for key in sorted(_tokenizer_vocab):
                _characters.update(key)
                self._vocab.add_token_to_namespace(key, "tokens")
        elif tokenizer is not None:
            _tokenizer_vocab = tokenizer.get_vocab()
            for key in sorted(_tokenizer_vocab):
                _characters.update(key)
                self._vocab.add_token_to_namespace(key, "tokens")
        self._vocab.add_tokens_to_namespace(_characters, "elmo_characters")

        # Text Field Embedder
        self._indexers = {
            "tokens": SingleIdTokenIndexer(),
            "elmo_characters": ELMoTokenCharactersIndexer()
        }

        _embedding = Embedding(
            embedding_dim=kwargs['embedding_dim'],
            num_embeddings=self._vocab.get_vocab_size("elmo_characters")
        )
        _encoder = CnnHighwayEncoder(
            embedding_dim=kwargs['embedding_dim'],
            filters=kwargs['filters'],
            num_highway=kwargs['enc_num_highway'],
            projection_dim=kwargs['enc_projection_dim'],
            activation=kwargs['enc_activation'],
            projection_location=kwargs['enc_projection_location'],
            do_layer_norm=kwargs['enc_do_layer_norm']
        )
        _characters_encoder = TokenCharactersEncoder(
            embedding=_embedding,
            encoder=_encoder
        )
        _text_field_embedder = BasicTextFieldEmbedder(
            token_embedders={"tokens": EmptyEmbedder(), "elmo_characters": _characters_encoder}
        )

        if kwargs['con_type'] == 'LSTM':
            _contextualizer = LstmSeq2SeqEncoder(
                input_size=kwargs['con_input_size'],
                hidden_size=kwargs['con_hidden_size'],
                num_layers=kwargs['con_num_layers'],
                bias=kwargs['con_bias'],
                dropout=kwargs['con_dropout'],
                bidirectional=kwargs['con_bidirectional'],
                stateful=kwargs['con_stateful']
            )
        elif kwargs['con_type'] == 'Transformer':
            _contextualizer = BidirectionalLanguageModelTransformer(
                input_dim=kwargs['con_input_size'],
                hidden_dim=kwargs['con_hidden_size'],
                num_layers=kwargs['con_num_layers'],
                dropout=kwargs['con_dropout'],
                input_dropout=kwargs['con_input_dropout'],
                return_all_layers=kwargs['con_return_all_layers']
            )
        else:
            raise ValueError('This is not valid value.'
                            ' You have to use "LSTM" or "Transformer".')


        # ELMo model.
        self.model = LanguageModel(
            vocab=self._vocab,
            text_field_embedder=_text_field_embedder,
            contextualizer=_contextualizer,
            dropout=kwargs['mod_dropout'],
            num_samples=kwargs['mod_num_samples'],
            sparse_embeddings=kwargs['mod_sparse_embeddings'],
            bidirectional=kwargs['mod_bidirectional']
        )

    def train(self, dataset):
        sentences = [dataset.get_line(idx) for idx in range(len(dataset))]
        instances = self._transform(sentences)
        
        train_loader = self._build_data_loader(instances)
        
        trainer = self._build_trainer(self.model, 10, 'tmp/elmo', train_loader)
        print("Starting training")
        trainer.train()
        print("Finished training")

    def save(self, serialization_dir):
        vocabulary_dir = os.path.join(serialization_dir, "vocabulary")
        weights_file = os.path.join(serialization_dir, "weights.th")

        self._vocab.save_to_files(vocabulary_dir)
        torch.save(self.model.state_dict(), weights_file)

    def load(self, serialization_dir):
        vocabulary_dir = os.path.join(serialization_dir, "vocabulary")
        weights_file = os.path.join(serialization_dir, "weights.th")
        self._vocab = Vocabulary.from_files(vocabulary_dir)
        self.model.load_state_dict(torch.load(weights_file))

    def vectorize(self, document, tokenizer=None, cuda=True):
        cuda_id = 1
        device = torch.device('cuda:{}'.format(cuda_id) if cuda else 'cpu')
        sentences = list()
        for sentence in document:
            sentence = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )["input_ids"].tolist()[0]
            sentence = tokenizer.convert_ids_to_tokens(sentence)
            sentences.append(sentence)
        instances = self._transform(sentences)
        batch = Batch(instances)
        batch.index_instances(self._vocab)
        self.model.to(device)
        model_input = move_to_device(batch.as_tensor_dict()['source'], device)
        output = self.model.forward(model_input)['lm_embeddings']
        output = output.cpu().detach().numpy()
        return np.mean(np.mean(output, axis=1), axis=0)

    def _transform(self, sentences):
        tokens = [[Token(t) for t in sent] for sent in sentences]
        return [Instance({'source': TextField(sent, self._indexers)}) for sent in tokens]

    def _build_data_loader(self, data):
        return SimpleDataLoader(data, 32, shuffle=True, vocab=self._vocab)

    def _build_trainer(self, model, nepoch, serialization_dir, data_loader, cuda=True):
        cuda_id = 1
        device = torch.device('cuda:{}'.format(cuda_id) if cuda else 'cpu')
        self.model.to(device)
        parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        optimizer = AdamOptimizer(parameters)
        trainer = GradientDescentTrainer(
            model=model,
            serialization_dir=serialization_dir,
            data_loader=data_loader,
            num_epochs=nepoch,
            optimizer=optimizer,
            cuda_device=cuda_id if cuda else -1
        )
        return trainer


if __name__ == '__main__':
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "data/tokenizers/byte_bpe-CSIC2010",
        max_len=512
    )
    dataset = CustomLineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="data/datasets/CSIC2010/training-normal",
        block_size=128,
    )
    model = Elmo(
        dataset=dataset,
        embedding_dim = 16,
        filters = [
                [1, 32],
                [2, 32],
                [3, 64],
                [4, 128],
                [5, 256],
                [6, 512],
                [7, 1024]
            ],
        enc_num_highway = 2,
        enc_projection_dim = 512,
        enc_activation = "relu",
        enc_projection_location = "after_highway",
        enc_do_layer_norm = True,
        con_type = 'LSTM',
        con_input_size = 512,
        con_hidden_size = 2048,
        con_num_layers = 2,
        con_bias = True,
        con_dropout = 0.0,
        con_bidirectional = True,
        con_stateful = False,
        con_input_dropout = None,
        con_return_all_layers = False,
        mod_dropout = 0.1,
        mod_num_samples = 8192,
        mod_sparse_embeddings = False,
        mod_bidirectional = True
    )
    model.train(dataset)
    model.save("tmp/elmo")
    
    model.load("tmp/elmo")
    with open("data/datasets/CSIC2010/testing-normal/1.txt") as f:
        document = f.readlines()
    array = model.vectorize(document, tokenizer)
    print(array)
    print(array.shape)
    print()