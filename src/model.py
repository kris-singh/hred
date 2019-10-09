#!/usr/bin/env python
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import GRU
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from config.model_config import cfg
from utils.vocab import create_vocab, prepare_data

class Model(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(Model, self).__init__()
        self.enc_hidden_sz = cfg.MODEL.ENC_HIDDEN_SZ
        self.ctx_hidden_sz = cfg.MODEL.CTX_HIDDEN_SZ
        self.dec_hidden_sz = cfg.MODEL.DEC_HIDDEN_SZ
        self.input_sz = cfg.MODEL.EMBED_SZ
        self.ctx_in_sz = cfg.MODEL.CTX_IN_SZ
        self.num_layers = cfg.MODEL.NUM_LAYERS

        self.embed = torch.nn.Embedding(kwargs['vocab_size'], cfg.MODEL.EMBED_SZ)

        self.enc = GRU(input_size=self.input_sz,
                       hidden_size=self.enc_hidden_sz,
                       num_layers=self.num_layers,
                       batch_first=True
                       )
        self.enc_to_ctx = nn.Linear(self.enc_hidden_sz, self.ctx_in_sz)
        self.ctx = GRU(input_size=self.ctx_in_sz,
                       hidden_size=self.ctx_hidden_sz,
                       num_layers=self.num_layers,
                       batch_first=True
                       )
        self.ctx_to_dec = nn.Linear(self.ctx_hidden_sz, self.dec_hidden_sz)
        self.dec = GRU(input_size=self.input_sz,
                       hidden_size=self.dec_hidden_sz,
                       num_layers=self.num_layers,
                       batch_first=True
                       )

    def forward(self, x, target):
        """
        Very important to understand the num_sent is batches here!
        x: Tensor of dim num_sent, seq_sz, embed_sz
        target: Tensor of dim num_sent, seq_sz, embed_sz
        """
        self.in_hid_encoder = torch.zeros(cfg.MODEL.NUM_LAYERS*cfg.MODEL.NUM_DIRECTIONS, x.shape[0], self.enc_hidden_sz)
        self.in_hid_ctx = torch.zeros(cfg.MODEL.NUM_LAYERS*cfg.MODEL.NUM_DIRECTIONS, x.shape[0], self.ctx_hidden_sz)
        x = self.embed(x)
        target = self.embed(target)
        # x: (batch, seq_sz, embed_sz)
        _, enc_hidden = self.enc(x, self.in_hid_encoder)
        # enc_hidden: (num_layers*num_directions*batch, enc_hidden_size)
        enc_hidden =enc_hidden.view(-1, self.enc_hidden_sz)
        ctx_in = self.enc_to_ctx(enc_hidden).view(cfg.MODEL.NUM_LAYERS*cfg.MODEL.NUM_DIRECTIONS, -1, self.ctx_in_sz)
        # ctx_in: (batch, num_layers*num_directions, enc_hidden_size)
        ctx_in = ctx_in.permute(1, 0, 2)
        # enc_hidden (batch, num_layers, ctx_in_sz)
        ctx_out, ctx_hidden = self.ctx(ctx_in, self.in_hid_ctx)
        ctx_hidden = ctx_hidden.view(-1, self.ctx_hidden_sz)
        dec_hidden = self.ctx_to_dec(ctx_hidden).view(cfg.MODEL.NUM_LAYERS*cfg.MODEL.NUM_DIRECTIONS, -1, self.dec_hidden_sz)
        # target (batch, seq_size, in_sz)
        dec_output, _ = self.dec(target, dec_hidden)


if __name__ == "__main__":
    # For testing model works!
    sent = ['mom i dont feel so good',
            'whats the issue',
            'i feel like i am going to pass out',
            ]
    vocab2idx, idx2vocab = create_vocab(sent)
    sent, target = prepare_data(sent, vocab2idx)
    model = Model(cfg, vocab_size=len(vocab2idx))
    model(sent, target)
