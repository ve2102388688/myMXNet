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
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        # (num_steps, batch_size)--->num_steps*(batch_size*vocab_size)
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # 先变成(num_steps*batch_size, num_hiddens)，之后output是(num_steps*batch_size, vocab_size)
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


################################################# TODO 训练模型 #####################################################
def predict_rnn_gluon(prefix, num_chars, model, vocab_size, ctx, idx_to_char, char_to_idx):
    # 使用model成员函数初始化隐藏层状态
    # FIXME state和X都需要在ctx上操作
    state = model.begin_state(batch_size=1, ctx=ctx)
    output = [char_to_idx[prefix[0]]]                   # TODO 给数组开个头,后面直接append,注意是[]类型

    for t in range(num_chars+len(prefix)-1):
        X = nd.array([output[-1]], ctx=ctx).reshape((1,1))
        (Y, state) = model(X, state)                    # TODO 前向计算时候不需要参数param
        # 预测最佳字符
        if t < len(prefix)-1:                           # 复制除第一个字剩下的prefix
            output.append(char_to_idx[prefix[t+1]])
        else:                                           # # Y[0]:1*1027,找出是1的那个下标,转成idx
            output.append(int(Y.argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])            

ctx = d2l.try_gpu()
model = RNNModel(rnn_layer, vocab_size)
model.initialize(force_reinit=True, ctx=ctx)
print(predict_rnn_gluon('分开', 10, model, vocab_size, ctx, idx_to_char, char_to_idx))



def train_and_predit_rnn_gluon(model, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes):
    loss = gloss.SoftmaxCrossEntropyLoss()
    model.initialize(force_reinit=True, ctx=ctx, init=init.Normal(sigma=0.01))
    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate':lr, 'momentum':0, 'wd':0})

    for epoch in range(num_epochs):
        l_sum, n, startTime = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx)
        state = model.begin_state(batch_size=batch_size)
        for X,Y in data_iter:
            for s in state:
                s.detach()
            with autograd.record():
                (output, state) = model(X, state)
                y = Y.T.reshape((-1, ))
                l = loss(output, y).meam()
            l.backward()

            params = [p.data() for p in model.collect_params().values()]
            d2l.grad_clipping(params, clipping_theta, ctx)
            Tra



















