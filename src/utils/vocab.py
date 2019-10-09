#!/usr/bin/env python

def create_vocab(sent):
    counter = Counter()
    vocab2idx = defaultdict()
    idx2vocab = defaultdict()

    for s in sent:
        words = s.split(' ')
        counter.update(words)

    for word in counter:
        vocab2idx[word] = len(vocab2idx)
        idx2vocab[vocab2idx[word]] = word

    return vocab2idx, idx2vocab

def prepare_data(sent, vocab2idx):
    for idx, s in enumerate(sent):
        s = s.split(' ')
        s = [vocab2idx[word] for word in s]
        s = torch.tensor(s, dtype=torch.long)
        sent[idx] = s

    sent = pad_sequence(sent).permute(1, 0)
    target = sent[1:]
    sent = sent[:-1]
    return sent, target
