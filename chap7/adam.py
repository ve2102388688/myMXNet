import d2lzh as d2l
from mxnet import nd


# 手动写一个
features, labels = d2l.get_data_ch7()

def inin_adam_states():
    v_w = nd.zeros((features.shape[1], 1))
    v_b = nd.zeros(1)
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p,(v, s) in zip(params, states):
        v[:] = beta1*v + (1-beta1)*p.grad
        s[:] = beta2*s + (1-beta2)*p.grad.square()
        corr_v = v/(1 - beta1 ** hyperparams['t'])
        corr_s = s/(1 - beta2 ** hyperparams['t'])
        g = (hyperparams['lr']*corr_v) / (corr_s.sqrt() + eps)
        p[:] -= g
    hyperparams['t'] += 1

d2l.plt.figure(figsize=(15,5))                 # 设置图片大小
# init_momentum_states()不要忘记括号，可能只是执行这个函数，而不是传进去
d2l.train_ch7(adam, inin_adam_states(), {'lr':0.01, 't':1}, features, labels)



# TODO 简洁实现
d2l.plt.figure(figsize=(15,5))                 # 设置图片大小
# 参数是rho，没有learning_rate!!
d2l.train_gluon_ch7('adam', {'learning_rate':0.01}, features, labels)


d2l.plt.show()


