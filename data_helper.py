# -*- coding: utf-8 -*-

import json
import collections
import random
import numpy as np


def load_json(file_path):
    with open(file_path, encoding='utf-8') as f:
        return json.loads(f.read())


def dump_json(obj, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)


def make_vocab(train_path, dev_path, test_path, vocab_path):
    words = ['<PAD>']
    counter = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            lis = line.strip().split('\t')
            counter.extend(list(lis[1]))

    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            lis = line.strip().split('\t')
            counter.extend(list(lis[1]))

    with open(dev_path, 'r', encoding='utf-8') as f:
        for line in f:
            lis = line.strip().split('\t')
            counter.extend(list(lis[1]))

    count = collections.Counter(counter).most_common(5999)

    for word, num in count:
        words.append(word)

    with open(vocab_path, 'w', encoding='utf-8') as f:
        for word in words:
            f.write(word+"\n")


def padding_data(data_path, word2id, label2id, max_seq_len):
    x = []
    y = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            lis = line.strip().split('\t')
            y.append(label2id[lis[0]])
            text = [word2id.get(w, 0) for w in list(lis[1].strip())] + [0]*max_seq_len
            x.append(text[:max_seq_len])
    random.seed(110)
    random.shuffle(x)
    random.seed(110)
    random.shuffle(y)
    return np.array(x), np.array(y)


def transfor_cnews(train_path, dev_path, test_path, vocab_path, word2id_path, label2id_path, max_seq_len):
    word2id = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for idx, word in enumerate(f):
            word2id[word.strip()] = idx

    dump_json(word2id, word2id_path)

    labels = set()
    with open(dev_path, 'r', encoding='utf-8') as f:
        for line in f:
            lis = line.strip().split('\t')
            labels.add(lis[0])

    label2id = {label: idx for idx, label in enumerate(list(labels))}

    dump_json(label2id, label2id_path)

    x_train, y_train = padding_data(train_path, word2id, label2id, max_seq_len)
    x_test, y_test = padding_data(test_path, word2id, label2id, max_seq_len)
    x_dev, y_dev = padding_data(dev_path, word2id, label2id, max_seq_len)

    return x_train, y_train, x_test, y_test, x_dev, y_dev


if __name__ == '__main__':
    train_data = "./data/cnews/cnews.train.txt"
    test_data = "./data/cnews/cnews.test.txt"
    dev_data = "./data/cnews/cnews.val.txt"
    vocab_data = "./data/cnews/cnews.vocab.txt"
    word2id_path = "./data/word2id.json"
    label2id_path = "./data/label2id.json"

    x_train_out = "./data/x_train.npy"
    y_train_out = "./data/y_train.npy"
    x_test_out = "./data/x_test.npy"
    y_test_out = "./data/y_test.npy"
    x_dev_out = "./data/x_dev.npy"
    y_dev_out = "./data/y_dev.npy"

    make_vocab(train_data, dev_data, test_data, vocab_data)

    x_train, y_train, x_test, y_test, x_dev, y_dev = transfor_cnews(train_data, dev_data, test_data, vocab_data,
                                                                    word2id_path, label2id_path, 100)

    np.save(x_train_out, x_train)
    np.save(y_train_out, y_train)
    np.save(x_test_out, x_test)
    np.save(y_test_out, y_test)
    np.save(x_dev_out, x_dev)
    np.save(y_dev_out, y_dev)


