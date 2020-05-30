import d2lzh as d2l
from mxnet import nd


"""**************************** 多输入通道 ****************************************"""
def corr2d_multi_in(X, K):
    # *对应位置相加配合add_n zip获取地址每一次取数据和对应的kernel
    return nd.add_n(*[d2l.corr2d(x,k) for x,k in zip(X, K)])

X = nd.array([[[0,1,2], [3,4,5], [6,7,8]],      # R-data
              [[1,2,3], [4,5,6],[7,8,9]]])      # G-data
K = nd.array([[[0,1], [2,3]],                   # R-kernel
               [[1,2], [3,4]]])                 # G-kernel
print(corr2d_multi_in(X, K))


"""**************************** 多输出通道 ****************************************"""
def corr2d_muti_in_out(X, K):
    # stack黏在后面　第一次(0 1 2 3)...
    # 遍历的是k的0通道的值,即第一对中括号中的逗号数目
    return nd.stack(*[corr2d_multi_in(X, k) for k in K])    # * 把2*2 2*2 2*2 弄成　[2*2 2*2 2*2]

# [(0 1 2 3) (1 2 3 4) (2 3 4 5)]
K = nd.stack(K, K+1, K+2)           # shape 3*2*2*2 [(2*2)(2*2) (2*2)(2*2) (2*2)(2*2)]
print(corr2d_muti_in_out(X, K))     # 输出三个　相当于corr2d_multi_in(X, K)　corr2d_multi_in(X, K＋１)　corr2d_multi_in(X, K＋２)


"""**************************** 1*1卷积层 ****************************************"""
# 核心: 每个通道数据先拉成行向量,运算,结果又拉回原形
def corr2d_multi_in_1x1(X, K):
    c_i, h, w = X.shape             # first-channel, for eaample, R,G,B(3*3*3)
    X = X.reshape((c_i, h*w))       # 将每一个通道的数据转成行向量, (3*9)
    c_o = K.shape[0]                # 输出多通道c_o
    K = K.reshape((c_o, c_i))       # 将对应的kernel转成行向量　(2*3)
    output = nd.dot(K, X)           # (2*3)*(3*9)=(2*9)
    return output.reshape((c_o, h, w))  # 将每一层数据拉回来(2*3*3)

X = nd.random.uniform(shape=(3, 3, 3))
K = nd.random.uniform(shape=(2, 3, 1, 1))
#　1*1卷积层可以当做宽高不变的全连接层
Y1 = corr2d_multi_in_1x1(X, K)
Y2 = corr2d_muti_in_out(X, K)
print('same?:', (Y1-Y2).norm().asscalar() < 1e-6)


