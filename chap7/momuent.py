import d2lzh as d2l
from mxnet import nd


# ################################################# TODO 梯度下降的问题 #####################################################
# # FIXME y = 0.2*x_1^2 + 2*x2^2
# # 不同方向上的梯度数量级相差大时，很难用一个确定的数来衡量，大一些将会发散，小一点某个方式上就动都不动

# eta = 0.4       # 0.6发散,0.4特别慢
# def fx_2d(x1, x2):              # f(x)
#     return 0.1*x1**2 + 2*x2**2

# def gd_2d(x1, x2, s1, s2):      # 梯度
#     return(x1-eta*0.2*x1, x2-eta*4*x2, 0, 0)

# d2l.plt.figure(figsize=(15,5))                 # 设置图片大小
# d2l.show_trace_2d(fx_2d, d2l.train_2d(gd_2d))


# ################################################# TODO 动量方法 #####################################################
# eta, gamma = 0.6, 0.3       # gamma(0-1)为0表示小批量随机梯度下降，值越大维持能力越强
# def momentum_2d(x1, x2, v1, v2):
#     v1 = gamma*v1 + 0.2*eta*x1
#     v2 = gamma*v2 + 4*eta*x2
#     return x1-v1, x2-v2, v1, v2

# d2l.plt.figure(figsize=(15,5))                 # 设置图片大小
# d2l.show_trace_2d(fx_2d, d2l.train_2d(momentum_2d))


################################################# TODO 动量方法 #####################################################
features, labels = d2l.get_data_ch7()

# def init_momentum_states():
#     v_w = nd.zeros((features.shape[1], 1))
#     v_b = nd.zeros(1)
#     return (v_w, v_b)

# def sgd_momentum(params, states, hyperparams):
#     for p,v in zip(params, states):
#         v[:] = hyperparams['momentum'] * v + hyperparams['lr'] * p.grad
#         p[:] -= v

# d2l.plt.figure(figsize=(15,5))                 # 设置图片大小
# # init_momentum_states()不要忘记括号，可能只是执行这个函数，而不是传进去
# # 可以认为批量增加多少，学习率减少多少，如momentum从0.5到0.9，即批量从2到10，增加5倍，学习率减少1/5，不然曲线不光滑
# d2l.train_ch7(sgd_momentum, init_momentum_states(), {'lr':0.004, 'momentum': 0.9}, features, labels)


# 简洁实现
d2l.plt.figure(figsize=(15,5))                 # 设置图片大小7
d2l.train_gluon_ch7('sgd', {'learning_rate':0.1, 'momentum':0.83}, features, labels)

d2l.plt.show()


