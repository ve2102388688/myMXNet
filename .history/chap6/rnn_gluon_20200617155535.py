import d2lzh as d2l
from mxnet import autograd, gluon, init
from mxnet.gluon import loss as gloss, nn
import math, time

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

################################################# TODO 定义模型 #####################################################
num_hiddens = 256
rnn_layer = rnn.RNN()







































