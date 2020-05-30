import mxnet as mx
from mxnet import nd
import time

# # 简单的展示gpu配置成功
# print(mx.cpu(), mx.gpu());
#
# # NDArray在CPU上运算
# x_cpu = nd.array([1, 2, 3]);
# print(x_cpu);                   # NDArray默认在CPU上 也就是物理内存上分配
# print(x_cpu.context);           # 通过context来查看NDArray所在的设备
#
#
# # NDArray在GPU上运算
# x_gpu = nd.array([1, 2, 3], ctx=mx.gpu());
# print(x_gpu);                   # NDArray默认在CPU上 也就是物理内存上分配
# print(x_gpu.context);           # 通过context来查看NDArray所在的设备



X = nd.random.normal(scale=10, shape=(5000000, 5000000))        # 0.00037384033203125
Y = nd.random.normal(scale=10, shape=(5000000, 5000000))
# X = nd.random.normal(scale=10, shape=(50000, 50000), ctx=mx.gpu())    # 3.5813279151916504
# Y = nd.random.normal(scale=10, shape=(50000, 50000), ctx=mx.gpu())
start = time.time()
for i in range(1000):
    Z = nd.dot(nd.linalg_inverse(X), nd.linalg_inverse(Y))
print(time.time()-start)
print(Z.context)

# X = nd.random.normal(scale=10, shape=(5000000, 5000000))
# start = time.time()
# Y = X. (mx.gpu())                                          # 2.8572134971618652
# print(time.time()-start)

