import d2lzh as d2l
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import math, time, numpy as np


# 读取数据
# corpus_indices 1w字的idx
# char_to_idx    字符转idx
# idx_to_char     idx转字符
# vocab_size不同字的个数
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

# one-hot向量
print(nd.one_hot(nd.array([1,2]), vocab_size))  # one-hot一行只有一个1,哪个位置是1呢？1,2位置

def to_onehot(X, size):
    return [nd.one_hot(x, size) for x in X.T]   # X中列是feature,行是sample

# Test
X = nd.arange(10).reshape((2, 5))   # 2:batch_size  5:num_step
inputs = to_onehot(X, vocab_size)   # 转成num_steps个形状为(batch_size,vocab_size)
np.set_printoptions(edgeitems=6)    # 显示个数设置,默认显示3个
print(len(inputs), inputs[0])       # 5个长度, 2*1027


################################################# TODO 初始化模型参数 #####################################################
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = d2l.try_gpu()
print('use ', ctx)
def get_param():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = nd.zeros(num_hiddens, ctx)
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx)

    # 求取梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params



#################################################  TODO 定义模型 #####################################################
# FIXME
# 隐藏状态　(批量大小, 隐藏层单元个数)
def init_rnn_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx), )       # 元祖表示不会改变数组,用圆括号表示

def rnn(inputs, state, params):
    # inputs和output都是num_steps个形状为(batch_size,vocab_size)
    output = []
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state              # 只有第一个有参数H,见上,shape=(batch_size, num_hiddens)
    for X in inputs:        # 遍历num_steps个
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)    # 计算隐藏状态,并作为返回值保存
        Y = nd.dot(H, W_hq) + b_q
        output.append(Y)    # 追加
    return output, (H, )

# 死一个时间步,输出形状,隐藏状态形状
state = init_rnn_state(X.shape[0], num_hiddens, ctx)    # (batch_size, num_hiddens)
inputs = to_onehot(X.as_in_context(ctx), vocab_size)    # 转成num_steps个形状为(batch_size,vocab_size)
params = get_param()
outputs, state_new = rnn(inputs, state, params)         # outputs,inputs一样的形状
print('len(outputs): ', len(outputs))                   # num_steps
print('outputs[0].shape: ', outputs[0].shape)           # (batch_size,vocab_size)
print('state_new[0].shape: ', state_new[0].shape)       # (batch_size, num_hiddens)



################################################# TODO 定义预测函数 #####################################################
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state, num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, ctx)        # 初始化H(state)
    outputs = [char_to_idx[prefix[0]]]                 # TODO 给数组开个头,后面直接append,注意是[]类型
    for t in range(num_chars+len(prefix)-1):           # 11次循环,头上面已经处理了
        # 处理输入，转成合法输入
        X = to_onehot(nd.array([outputs[-1]], ctx=ctx), vocab_size)     #　上一步的输出作为当前的输入
        # 计算输出,更新状态
        (Y, state) = rnn(X, state, params)           
        # 预测最佳字符                   
        if t < len(prefix)-1:                          # 复制除第一个字剩下的prefix
            outputs.append(char_to_idx[prefix[t+1]])
        else:                                          # Y[0]:1*1027,找出是1的那个下标,转成idx
            outputs.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in outputs])  # 将idx转换成char,即运算的时候全是int,在开头结尾进行转换

# 预测以'分开'的接下来10字,孙吉初始化输出随机
predict = predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx)
print('predict: ', predict)


# TODO 裁剪梯度　防止梯度衰减或爆炸
def grad_clipping(params, theta, ctx):
    norm = nd.array([0], ctx=ctx)
    for param in params:
        norm += (param.grad()**2).sum()
    norm = norm.sqrt().asscalar()       # norm means g
    if theta < norm:                    # min(theta/g, 1)*g
        for param in params:
            param.grad[:] *= theta/norm


################################################# TODO 定义模型训练函数 #####################################################
def train_and_predit_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, ctx, corpus_indices,
                         idx_to_char, char_to_idx, is_random_iter, num_epochs, num_steps, lr, clipping_theta,
                         batch_size, pred_period, pred_len, prefixes):
    # 采样方式
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random             # 随机取样
    else:
        data_iter_fn = d2l.data_iter_consecutive        # 相邻取样

    params = get_param()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:                                      # 如果是相邻采样，刚开始就初始化隐藏层的参数
            state = init_rnn_state(batch_size, num_hiddens, ctx)

        # 读取数据
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X,Y in data_iter:
            if is_random_iter:                                      # 如果是随机采样，在小批量更新前初始化隐藏层的参数
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:
                for s in state:                                     # 否则从计算图中分离出来
                    s.detach()
            with autograd.record():
                inputs = to_onehot(X, vocab_size)
                (outputs, state) = rnn(inputs, state, params)       # inputs和outputs是num_steps个(batch_size, vocab_size)
                outputs = nd.concat(*outputs, dim=0)                # 联结之后形状为(batch_size*num_steps, vocab_size)
                y = Y.T.reshape((-1, ))                             # Y(batch_size, num_steps),转置后拉成成为行向量
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx)              # 剪裁梯度
            d2l.sgd(params, lr, 1)                                  # 里面不用填
            l_sum += l.asscalar() * y.size
            n += y.size

        if (epoch+1)%pred_period == 0:
            print('epoch %d, perplexity %f, time %.3f sec' % (epoch+1, math.exp(l_sum/n), time.time()-time))
            for prefix in prefixes:
                print('-', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state, num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))


num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 100, 0.01
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

train_and_predit_rnn(rnn, get_param, init_rnn_state, num_hiddens, vocab_size, ctx, corpus_indices,
                     idx_to_char, char_to_idx, True, num_epochs, num_steps, lr, clipping_theta,
                     batch_size, pred_period, pred_len, prefixes)

