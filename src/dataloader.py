#!/usr/bin/env python

import csv
import json
import re
from collections import Counter, OrderedDict, defaultdict

import nltk
from torch.utils.data import Dataset


def read_movies():
    idx2movie = defaultdict()
    with open('../data_small/movies_with_mentions.csv') as f:
        reader = csv.reader(f)
        date_pattern = re.compile(r'\(\d{4}\)')
        idx2movie= {row[0]: date_pattern.sub('', row[1]) for row in reader if row[0] != "movieId"}
    return idx2movie


def sub_movies(conversation):
    pattern = re.compile(r'@(\d+)')
    idx2movie = read_movies()
    for cid, conv in conversation.items():
        for sid, sent in conv.items():
            matches = pattern.finditer(sent)
            for match in matches:
                start, end, movie_idx = match.start(), match.end(), match.group(0)
                movie_name = idx2movie[movie_idx[1:]]
                sent = re.sub(movie_idx, ' ' + movie_idx + ' ', sent)
            conversation[cid][sid] = sent
    return conversation

def create_vocab(conversation):
    counter = Counter()
    vocab2idx, idx2vocab = defaultdict(), defaultdict()

    markers = [('<SOS>', 0), ('<EOS>', 1), ('<UNK>', 2), ('<MVN>', 3)]
    for mark in markers:
        vocab2idx[mark[0]], idx2vocab[mark[1]] = mark[1], mark[0]
    for cid, conv in conversation.items():
        for sid, sent in conv.items():
            sent = sent.lower()
            words = nltk.tokenize.word_tokenize(sent)
            conversation[cid][sid] = ['<SOS>'] + words + ['<EOS>']
            counter.update(words)
    counter.most_common(15000)
    for word in counter:
        vocab2idx[word] = len(vocab2idx)
        idx2vocab[vocab2idx[word]] = word

    return vocab2idx, idx2vocab, conversation


def load_train(path):
    with open(path) as f:
        train_data = f.read()

    train_data = train_data.split('\n')
    conversation = defaultdict(lambda: defaultdict())
    for td in train_data[:-1]:
        content = json.loads(td)
        messages = content['messages']
        conversationId = content['conversationId']
        prev = -1
        for msg in messages:
            mid = msg['messageId']
            if prev == msg['senderWorkerId']:
                conversation[conversationId][prev_mid] = conversation[conversationId][prev_mid][-1] + msg['text']
            else:
                conversation[conversationId][mid] = msg['text']
                prev_mid = mid
            prev = msg['senderWorkerId']

        for cid, conv in conversation.items():
            conversation[cid] = dict(sorted(conv.items()))


    vocab2idx, idx2vocab, conversation = create_vocab(conversation)
    conversation = sub_movies(conversation)

    data = defaultdict(lambda: defaultdict(lambda: defaultdict()))
    min_key = min(conversation.keys())
    for cid,conv in conversation.items():
        did = int(cid) - int(min_key)
        for sid, sent in conv.items():
            words = nltk.tokenize.word_tokenize(sent)
            sent = [0]
            for word in words:
                if word in vocab2idx.keys():
                    sent.append(vocab2idx[word])
                else:
                    print(word)
                    sent.append(vocab2idx['<UNK>'])
            sent.append(1)
            data[did][sid]['text'] = sent
            data[did][sid]['length'] = len(sent)
        data[did]['conv_length'] = len(conv)

    CONV_LIMIT = 10
    SENT_LIMIT = 30
    for cid, conv in data.items():
        if conv['conv_length'] > CONV_LIMIT:
            del_keys = list(data[cid].keys())[CONV_LIMIT:]
            for dk in del_keys:
                data[cid].pop(dk)
            data[cid]['conv_length'] = CONV_LIMIT

    for cid, conv in data.items():
        for sid, sent in conv.items():
            if sid=='conv_length':
                continue
            if data[cid][sid]['length'] > SENT_LIMIT:
                data[cid][sid]['text'] = data[cid][sid]['text'][:SENT_LIMIT]
                data[cid][sid]['length'] = SENT_LIMIT

    return data, vocab2idx, idx2vocab

class RecDialDataloader(Dataset):

    def __init__(self, path):
        self.train_path = path['train']
        # load dataset
        self.train, self.vocab2idx, self.idx2vocab = load_train(self.train_path)

    def __getitem__(self, idx):
       return self.train[idx]

    def __len__(self):
       return len(self.train)

if __name__ == '__main__':
    path = defaultdict()
    path['train'] = '../data_small/train_data_small.jsonl'

    data = RecDialDataloader(path)
    for sid, sent in data[0].items():
        print(sent)
        for word in sent['text']:
            print(data.idx2vocab[word], end=' ')
