import d2lzh as d2l
import math
from mxnet import nd


# FIXME y = 0.2*x_1^2 + 2*x2^2　测试
eta = 2         # 比之前的大，但是效果很好，不会发散
def adagrad_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2*x1, 4*x2, 1e-6
    s1 += g1**2
    s2 += g2**2
    x1 -= eta/math.sqrt(s1+eps) * g1
    x2 -= eta/math.sqrt(s2+eps) * g2
    return x1, x2, s1, s2

def fx(x1, x2):
    return 0.1*x1**2 + 2*x2**2

d2l.plt.figure(figsize=(15,5))                 # 设置图片大小7
d2l.show_trace_2d(fx, d2l.train_2d(adagrad_2d))



# 手动写一个
features, labels = d2l.get_data_ch7()
def inin_adagrad_states():
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p,s in zip(params, states):
        s[:] += p.grad.square()
        p[:] -= hyperparams['lr'] * p.grad / (s+eps).sqrt()


d2l.plt.figure(figsize=(15,5))                 # 设置图片大小
# init_momentum_states()不要忘记括号，可能只是执行这个函数，而不是传进去
# 可以认为批量增加多少，学习率减少多少，如momentum从0.5到0.9，即批量从2到10，增加5倍，学习率减少1/5，不然曲线不光滑
d2l.train_ch7(adagrad, inin_adagrad_states(), {'lr':0.1}, features, labels)



# TODO 简洁实现
d2l.plt.figure(figsize=(15,5))                 # 设置图片大小7
d2l.train_gluon_ch7('adagrad', {'learning_rate':0.1}, features, labels)


d2l.plt.show()


