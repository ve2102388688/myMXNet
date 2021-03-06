import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, rnn
import math, time

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

################################################# TODO 定义模型 #####################################################
# FIXME 1.创建一个256的rnn 2.给出初始状态　3.丢进网络一次测试
# 构造单隐藏层，个数是256
num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()

batch_size = 2
state = rnn_layer.begin_state(batch_size=batch_size)        # 初始值
print(state[0].shape)    # （隐藏层个数，batch_size，num_hiddens）

num_steps = 35      # 34元语法
X = nd.random.uniform(shape=(num_steps, batch_size, vocab_size))
Y, state_new = rnn_layer(X, state)                          # 把数据X和初始状态state丢进网络rnn_layer里
print(Y.shape, len(state_new), state_new[0].shape)          # state_new和state同型

# 定义一个完整的循环神经网络
class RNNModel(nn.Block):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.rnn = rnn_layerforce_reinit
    # TODO 1.转换成合适输入格式　2.丢进网络　3.进全连接层
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
# FIXME 1.隐藏层的初始化　2.前向计算
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

# Test
ctx = d2l.try_gpu()
model = RNNModel(rnn_layer, vocab_size)             # 定义模型
model.initialize(force_reinit=True, ctx=ctx)        # reinit
print(predict_rnn_gluon('分开', 10, model, vocab_size, ctx, idx_to_char, char_to_idx))

################################################# TODO 训练模型 #####################################################
# FIXME 1.训练　2.调用上面的预测
def train_and_predit_rnn_gluon(model, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes):
    # 定义损失函数　模型初始化　优化函数
    loss = gloss.SoftmaxCrossEntropyLoss()
    model.initialize(force_reinit=True, ctx=ctx, init=init.Normal(sigma=0.01))
    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate':lr, 'momentum':0, 'wd':0})

    for epoch in range(num_epochs):
        l_sum, n, startTime = 0.0, 0, time.time()
        # get batch of train_data and init
        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx)
        state = model.begin_state(batch_size=batch_size, ctx=ctx)

        for X,Y in data_iter:
            for s in state:     # TODO 从计算图分离，减少梯度计算随次数的增加而增加
                s.detach()
            with autograd.record():
                (output, state) = model(X, state)
                y = Y.T.reshape((-1, ))                 # Y(batch_size, num_steps),转置后拉成成为行向量
                l = loss(output, y).mean()              # 计算平均分类损失
            l.backward()

            # 剪裁梯度
            params = [p.data() for p in model.collect_params().values()]
            d2l.grad_clipping(params, clipping_theta, ctx)
            trainer.step(1)                             # 前进一步
            l_sum += l.asscalar() * y.size
            n += y.size

        if (epoch+1)%pred_period == 0:  # 每pred_period打印一次
            # TODO math.exp(l_sum/n)困惑度
            print('epochs %d, perplexity %f, time %0.2f' % (epoch+1, math.exp(l_sum/n), time.time()-startTime))
            for prefix in prefixes:
                print('-', predict_rnn_gluon(prefix, pred_len, model, vocab_size, ctx, idx_to_char, char_to_idx))


# Test
num_epochs, batch_size, lr, clipping_theta = 250, 32, 100, 0.01
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开'] 
train_and_predit_rnn_gluon(model, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char, char_to_idx, 
                            num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes)



