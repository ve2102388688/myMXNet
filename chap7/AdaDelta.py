import d2lzh as d2l
from mxnet import nd


# 手动写一个
features, labels = d2l.get_data_ch7()

def inin_adadelta_states():
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    delta_w = nd.zeros((features.shape[1], 1))
    delta_b = nd.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p,(s, delta) in zip(params, states):
        s[:] = rho*s + (1-rho) * p.grad.square()                # 更新状态s
        g = ((delta+eps).sqrt() / (s+eps).sqrt()) * p.grad      # 更新params
        p[:] -= g
        delta[:] = rho*delta + (1-rho)*g*g                      # 更新delta

d2l.plt.figure(figsize=(15,5))                 # 设置图片大小
# init_momentum_states()不要忘记括号，可能只是执行这个函数，而不是传进去
d2l.train_ch7(adadelta, inin_adadelta_states(), {'rho':0.9}, features, labels)



# TODO 简洁实现
d2l.plt.figure(figsize=(15,5))                 # 设置图片大小
# 参数是rho，没有learning_rate!!
d2l.train_gluon_ch7('adadelta', {'rho':0.1}, features, labels)


d2l.plt.show()


