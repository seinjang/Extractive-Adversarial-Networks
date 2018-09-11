import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
# from gumbel import Gumbel

torch.manual_seed(226)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model(vocab, args, snapshot):
    # initialize a new model
    print("\nBuilding model...")
    G = Generator(vocab, args)
    P_1 = Primary_Predictor(vocab, args)
    P_2 = Adversarial_Predictor(vocab, args)

    print("Total num. of parameters: %d" % (count_parameters(G)+count_parameters(P_1)+count_parameters(P_2)))

    models = {}
    models['G'] = G.cuda()
    models['P_1'] = P_1.cuda()
    models['P_2'] = P_2.cuda()

    if args.cuda:
        return models
    else:
        return models

class Embedding(nn.Module):
    def __init__(self, vocab, fine_tune_wv):
        '''
            This module aims to convert the token id into its corresponding embedding.
        '''
        super(Embedding, self).__init__()
        vocab_size, embedding_dim = vocab.vectors.size()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_layer.weight.data = vocab.vectors
        self.embedding_layer.weight.requires_grad = False
        # do not finetune the embedding

        self.fine_tune_wv = fine_tune_wv
        if self.fine_tune_wv > 0:
            # if strictly positive, augment the original fixed embedding by a tunable embedding of
            # dimension fine_tune_wv
            self.tune_embedding_layer = nn.Embedding(vocab_size, self.fine_tune_wv)

        self.embedding_dim = embedding_dim + fine_tune_wv

    def forward(self, text):
        '''
            Argument:
                text:   batch_size * max_text_len
            Return:
                output: batch_size * max_text_len * embedding_dim
        '''
        output = self.embedding_layer(text).float()

        if self.fine_tune_wv > 0:
            output = torch.cat([output, self.tune_embedding_layer(text).float()], dim=2)

        return output

class RNN(nn.Module):
    def __init__(self, cell_type, input_dim, hidden_dim, num_layers, bidirectional, dropout):
        '''
            This module is a wrapper of the RNN module
        '''
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                    bidirectional=bidirectional, dropout=dropout)


    def _sort_tensor(self, input, lengths):
        sorted_lengths, sorted_order = lengths.sort(0, descending=True)
        sorted_input = input[sorted_order]
        _, invert_order  = sorted_order.sort(0, descending=False)

        # Calculate the num. of sequences that have len 0
        nonzero_idx = sorted_lengths.nonzero()
        num_nonzero = nonzero_idx.size()[0]
        num_zero = sorted_lengths.size()[0] - num_nonzero

        # temporarily remove seq with len zero
        sorted_input = sorted_input[:num_nonzero]
        sorted_lengths = sorted_lengths[:num_nonzero]

        return sorted_input, sorted_lengths, invert_order, num_zero

    def _unsort_tensor(self, input, invert_order, num_zero):
        if num_zero == 0:
            input = input.index_select(0, invert_order)

        else:
            dim0, dim1, dim2 = input.size()
            zero = torch.zeros(num_zero, dim1, dim2)
            if self.args.cuda:
                zero = zero.cuda()

            input = torch.cat((input, zero), dim=0)
            input = input[invert_order]

        return input

    def forward(self, text, text_len):
        # Go through the rnn
        # Sort the word tensor according to the sentence length, and pack them together
        sort_text, sort_len, invert_order, num_zero = self._sort_tensor(input=text, lengths=text_len)
        # print(sort_len.cpu().numpy().astype(int))
        text = pack_padded_sequence(sort_text, lengths=sort_len.cpu().numpy().astype(int), batch_first=True)

        # Run through the word level RNN
        text, _ = self.rnn(text)         # batch_size, max_doc_len, args.word_hidden_size

        # Unpack the output, and invert the sorting
        text = pad_packed_sequence(text, batch_first=True)[0] # batch_size, max_doc_len, rnn_size
        text = self._unsort_tensor(text, invert_order, num_zero) # batch_size, max_doc_len, rnn_size

        return text

class Generator(nn.Module):

    def __init__(self, vocab, args):
        super(Generator, self).__init__()
        self.args = args

        # Word embedding
        self.ebd = Embedding(vocab, args.fine_tune_wv)

        # Generator RNN
        self.rnn = RNN(args.cell_type, self.ebd.embedding_dim , args.rnn_size//2, 1, True,
                args.dropout)

        self.gen_fc = nn.Linear(args.rnn_size, 2)

        # Fully connected
        if args.hidden_dim != 0:
            self.seq = nn.Sequential(nn.Sigmoid(), nn.Dropout(args.dropout))
            """
            self.seq = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(self.num_filters_total, args.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(args.hidden_dim, args.num_classes)
                    )
            """

        else:
            self.seq = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(args.num_filters_total, args.num_classes)
                    )

        self.dropout = nn.Dropout(args.dropout)

    def _gumbel_softmax(self, logit, temperature, cuda):
        '''
        generate a gumbel softmax based on the logit
        noise is a sample from gumbel(0,1)
        '''
        eps = 1e-20
        noise = torch.rand(logit.size()).cuda()
        noise = - torch.log(-torch.log(noise+eps)+eps)
        x = (logit + noise.detach()) / temperature
        return F.softmax(x, dim=-1)


    def _independent_sampling(self, x, temperature, hard):
        '''
        Use the hidden state at all time to sample whether each word is a rationale or not.
        No dependency between actions.
        Return the sampled soft rationale mask
        '''
        rationale_logit = F.log_softmax(self.gen_fc(x), dim=2)
        rationale_mask  = self._gumbel_softmax(rationale_logit, temperature, self.args.cuda)

        # extract the probability of being a rationale from the two dimensional gumbel softmax
        rationale_mask  = rationale_mask[:,:,1]

        if hard: # replace soft mask by hard mask, no longer differentiable, only used during testing
            rationale_mask = torch.round(rationale_mask).float()

        return rationale_mask


    def forward(self, text, text_len, temperature, hard=False):
        word = self.ebd(text)
        word = self.dropout(word)

        # Generator
        G = self.rnn(word, text_len)


        # Sample rationale indicator
        rationale = self._independent_sampling(G, temperature, hard)

        # mask out non-rationale words
        rat_word = word * rationale.unsqueeze(2)  # batch, len, embedding_dim

        # run the MLP
        out = rat_word

        return out, rationale


class Primary_Predictor(nn.Module):

    def __init__(self, vocab, args):
        super(Primary_Predictor, self).__init__()
        self.args = args

        # Predictor_1 RNN
        self.rnn = RNN(args.cell_type, 300, args.rnn_size//2, 1, True,
                args.dropout)


        # final sigmoid output
        # self.seq = nn.Sequential(nn.Sigmoid(), nn.Dropout(args.dropout))

        self.seq = nn.Sequential(
                nn.ReLU(),
                nn.Linear(300, args.hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_dim, args.num_classes)
                )
        self.dropout = nn.Dropout(args.dropout)


    def forward(self, text, text_len, temperature, hard=False):

        # Predictor_1
        P_1 = self.rnn(text, text_len)


        out = self.seq(P_1[:,-1])

        return out

class Adversarial_Predictor(nn.Module):

    def __init__(self, vocab, args):
        super(Adversarial_Predictor, self).__init__()
        self.args = args

        # Predictor_2 RNN
        self.rnn = RNN(args.cell_type, 300, args.rnn_size//2, 1, True,
                args.dropout)


        # final sigmoid output
        # self.seq = nn.Sequential(nn.Sigmoid(), nn.Dropout(args.dropout))

        self.seq = nn.Sequential(
                nn.ReLU(),
                nn.Linear(300, args.hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_dim, args.num_classes)
                )
        self.dropout = nn.Dropout(args.dropout)


    def forward(self, text, text_len, temperature, hard=False):

        # Predictor_1
        P_2 = self.rnn(text, text_len)

        out = self.seq(P_2[:,-1])

        return out
