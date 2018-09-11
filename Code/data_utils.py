import numpy as np
import re
import torch
import pickle
import itertools
import collections
import random
import json
import os
import csv
import gzip
from torchtext.vocab import Vocab
from colored import fg, attr, bg
import textwrap
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

def load_data(path, regression):
    '''
    path: path to the tsv file which contains just label and text
    '''
    data_len = []
    label = []

    with open(path, 'r', encoding='utf8') as f:
        next(f)
        data = []

        for line in f:
            line = line.strip().split('\t')
            if len(line) == 3:
                data.append({
                    'label': float(line[1]) if regression else int(float(line[1])),
                    'text': line[2].split()
                    })

                label.append(data[-1]['label'])
                data_len.append(len(data[-1]['text']))

        counter = collections.Counter(data_len)
        print('Average length: ' + str(sum(data_len)/len(data_len)))
        label = np.array(label)
        print(collections.Counter(label))
        # print("Label mean: %.4f, variance: %.4f" % (np.mean(label), np.var(label)))

    return data

def load_dataset_att(args):
    # aspect = int(args.dataset[-1])

    if args.mode == 'train':
        train_data = load_data('../Dataset/attack.train', regression=True)
        dev_data   = load_data('../Dataset/attack.dev', regression=True)
        test_data  = []

    elif args.mode == 'test':
        train_data = []
        dev_data   = []
        test_data  = load_data(args.test_path, regression=True)

    else:
        raise ValueError('Mode can only be train, test')

    return train_data, dev_data, test_data

def read_words(data):
    words = []
    for example in data:
        words += example['text']
    return words

def data_to_nparray(data, vocab, args):
    '''
        Convert the data into a dictionary of np arrays for speed.
    '''
    # process the text
    text_len = np.array([len(e['text']) for e in data])
    max_text_len = max(text_len)

    text = np.ones([len(data), max_text_len], dtype=np.int64) * vocab.stoi['<pad>']
    for i in range(len(data)):
        text[i,:len(data[i]['text'])] = [
                vocab.stoi[x] if x in vocab.stoi else vocab.stoi['<unk>'] for x in data[i]['text']]

    if args.num_classes == 1: # regression task
        doc_label = np.array([x['label'] for x in data], dtype=np.float32)

    else: # classification task
        doc_label = np.array([x['label'] for x in data], dtype=np.int64)

    raw = np.array([e['text'] for e in data], dtype=object)

    return {'text':      text,
            'text_len':  text_len,
            'label':     doc_label,
            'raw':       raw,
            }

def load_dataset(args, vocab=None):
    print("Loading data...")
    train_data, dev_data, test_data = load_dataset_att(args)

    if vocab is None:
        vocab = Vocab(collections.Counter(read_words(train_data)), vectors=args.word_vector)

    wv_size = vocab.vectors.size()
    print('Total num. of words: %d\nWord vector dimension: %d' % (wv_size[0], wv_size[1]))

    num_oov = wv_size[0] - torch.nonzero(torch.sum(torch.abs(vocab.vectors), dim=1)).size()[0]
    print('Num. of out-of-vocabulary words (they are initialized to zero vector): %d'\
            % num_oov)

    print('Length of data: train: %d, dev: %d, test: %d.' % (len(train_data), len(dev_data), len(test_data)))

    if args.mode == 'train':
        train_data = data_to_nparray(train_data, vocab, args)
        dev_data   = data_to_nparray(dev_data, vocab, args)
    elif args.mode == 'test':
        test_data  = data_to_nparray(test_data, vocab, args)
    else:
        raise ValueError('Invalid mode')

    return train_data, dev_data, test_data, vocab

def data_loader(data, batch_size, shuffle=True):
    """
        Generates a batch iterator for a dataset.
    """
    data_size = len(data['label'])
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1

    np.random.seed(1)

    # shuffle the dataset at the begging of each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        for key, value in data.items():
            data[key] = value[shuffle_indices]

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)

        max_text_len = max(data['text_len'][start_index:end_index])

        yield (data['text'][start_index:end_index,:max_text_len],
                data['text_len'][start_index:end_index],
                data['label'][start_index:end_index],
                data['raw'][start_index:end_index])

def generate_writer(path, refilter=False):
    writer = {}

    if not refilter:
        writer['human'] = open(path + '.human_readable.tsv', 'w', encoding='utf8')
        writer['machine'] = open(path + '.machine_readable.tsv', 'w', encoding='utf8')
        writer['filtered_human'] = open(path + '.human_readable.filtered.tsv', 'w', encoding='utf8')
        writer['filtered_machine'] = open(path + '.machine_readable.filtered.tsv', 'w', encoding='utf8')

        writer['human'].write("task\ttrue_lbl\tpred_lbl\ttext\n")
        writer['machine'].write("task\tlabel\ttext\trationale\n")
        writer['filtered_human'].write("task\ttrue_lbl\tpred_lbl\ttext\n")
        writer['filtered_machine'].write("task\tlabel\ttext\trationale\n")

    return writer

def close_writer(writer):
    for key in writer.keys():
        writer[key].close()
