#!/usr/bin/env python
import os
import csv
import json
import re
from collections import Counter, OrderedDict, defaultdict

import ipdb
import nltk
from torch.utils.data import Dataset

sep = " +++$+++ "
delimiter = "\n"


def create_vocab(conversation):
    counter = Counter()
    vocab2idx, idx2vocab = defaultdict(), defaultdict()

    markers = [('<SOS>', 0), ('<EOS>', 1), ('<UNK>', 2)]
    for mark in markers:
        vocab2idx[mark[0]], idx2vocab[mark[1]] = mark[1], mark[0]
    for conv, conv_len in conversation:
        for sent, sent_len in conv:
            for words  in sent:
                counter.update(words)
    counter.most_common(15000)
    for word in counter:
        vocab2idx[word] = len(vocab2idx)
        idx2vocab[vocab2idx[word]] = word
    return vocab2idx, idx2vocab, conversation


def load_lines(path, utterances):
    lines_dict = defaultdict()
    conv = []
    lines_path = os.path.join(path, "movie_lines.txt")
    with open(lines_path, 'r', encoding='iso-8859-1') as f:
        lines_data = f.read().split('\n')[:-1]

    for lines in lines_data:
        lines = lines.split(sep)
        lines_dict[lines[0]] = (lines[1], lines[-1])

    prev_id = None
    for utter in utterances:
        id, line = lines_dict[utter]
        line = line.lower()
        line = nltk.tokenize.word_tokenize(line)
        line = ['<SOS>'] + line + ['<EOS>']
        if prev_id == id:
            conv[-1][0] += line
            conv[-1][1] += len(line)
        else:
            conv.append((line, len(line)))
        prev_id = id

    conv_length = len(utterances)
    conversation = (conv, conv_length)
    return conversation


def create_data(path):
    conversation_path = os.path.join(path, "movie_conversations.txt")
    with open(conversation_path) as f:
        conversations_data = f.read().split('\n')

    conv_data = []
    for conversation in conversations_data[:-1]:
        utterances = eval(conversation.split(sep)[3])
        conv  = load_lines(path, utterances)
        conv_data.append(conv)

    vocab2idx, idx2vocab, conversation = create_vocab(conv_data)

    CONV_LIMIT = 10
    SENT_LIMIT = 30
    del_keys = []
    for id, (conv, conv_len) in enumerate(conv_data):
        if conv_len > CONV_LIMIT:
            del_keys.append(id)

    for keys in del_keys:
        conv.pop(keys)

    for _, (conv, conv_len) in enumerate(conv_data):
        for id, (sent, sent_len) in enumerate(conv):
            if sent_len > SENT_LIMIT:
                sent = sent[:SENT_LIMIT]
                conv[id] = (sent, SENT_LIMIT)

    return conv_data, vocab2idx, idx2vocab


def create_splits(splits, path):
    data, vocab2idx, idx2vocab = create_data(path)
    train_idx = int(len(data) * splits['train'])
    val_idx = int(len(data) * splits['val'])
    train = data[0 : train_idx]
    val = data[train_idx: train_idx + val_idx]
    test = data[train_idx+val_idx:]
    return train, val, test, vocab2idx, idx2vocab


if __name__ == '__main__':
    path = '../data_small'
    splits = defaultdict()
    splits['train'] = 0.7
    splits['val'] = 0.2
    splits['test'] = 0.1
    train, val, test, vocab2idx, idx2vocab = create_splits(splits, path)
