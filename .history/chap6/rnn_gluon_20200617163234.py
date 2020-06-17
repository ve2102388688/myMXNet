import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, rnn
import math, time

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

################################################# TODO 定义模型 #####################################################
# 构造单隐藏层，个数是256
num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()

batch_size = 2
state = rnn_layer.begin_state(batch_size=batch_size)        # 返回初始化隐藏状态列表
print(state[0].shape)    # （隐藏层个数，batch_size，num_hiddens）


num_steps = 35
X = nd.random.uniform(shape=(num_steps, batch_size, vocab_size))
Y, state_new = rnn_layer(X, state)
print(Y.shape, len(state_new), state_new[0].shape)          # state_new和state同型


class RNNModel(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs):
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense()






























