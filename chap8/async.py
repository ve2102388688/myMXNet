from mxnet import autograd, gluon, nd
from mxnet.gluon import loss as gloss, nn
import subprocess, time, os

# 异步计算
class Benchmark():
    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''
    
    def __enter__(self):            # 进入开始计时
        self.start = time.time()
    
    def __exit__(self, *args):      # 退出计算耗时
        print('%stime:%fsec' % (self.prefix, time.time()-self.start))

# 前端把任务丢到后端的线程队列上，就返回了，耗时很少
with Benchmark('Workloads are quened.'):        # 0.019261sec
    x = nd.random.uniform(shape=(2000,2000))
    y = nd.dot(x, x).sum()

with Benchmark('Workloads are finished.'):      # 0.118694sec
    print('sum:', y)    # 同步函数，前端后等后端的结果，真正的计算


# wait_to_read waitall asscalar asnumpy都是同步函数，前端一直等后端出结果
with Benchmark('Sigle task'):           # time:0.075136sec
    y = nd.dot(x, x)
    y.wait_to_read()                    # 等单个NDArray

with Benchmark('Two task'):             # time:0.145045sec
    y = nd.dot(x, x)
    z = nd.dot(x, x)
    nd.waitall()                        # 等所有NDArray

with Benchmark():           # time:0.075136sec
    y = nd.dot(x, x)
    y.asnumpy()

with Benchmark():           #　time:0.121387sec
    y = nd.dot(x, x)
    y.norm().asscalar()


# 下面两个看不出啥区别，可能是任务太简单了,被优化了
with Benchmark('synch'):
    for _ in range(1000):
        y = x + 1
        y.wait_to_read()    # 同步计算

with Benchmark('asynch'):
    for _ in range(1000):
        y = x + 1
    nd.waitall()            # 异步计算


# 看一个大点的的例子
def data_iter():
    start = time.time()
    num_batches, batch_size = 100, 1024
    for i in range(num_batches):
        X = nd.random.normal(shape=(batch_size, 512))
        y = nd.ones((batch_size, ))
        yield X, y                              # 相当于return,下次运行会从这里开始
        if (i+1) % 50 == 0:
            print('batch %d, time %f sec' % (i+1, time.time()-start))

# 定义一个简单的网络测试
net = nn.Sequential()
net.add(nn.Dense(2048, activation='relu'), 
        nn.Dense(512, activation='relu'), 
        nn.Dense(1))
net.initialize()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.005})
loss = gloss.L2Loss()


# USER PID CPU MEM VSZ
def get_mem():
    res = subprocess.check_output(['ps', 'u', '-p', str(os.getpid())])
    return int(str(res).split()[15]) / 1000     # 获取该进程的虚拟内存大小VSZ,单位Mb

for X,y in data_iter():
    break
# loss(y, net(X)).wait_to_read()

# 同步计算
l_sum, mem = 0, get_mem()
for X,y in data_iter():
    with autograd.record():
        l = loss(net(X), y)
    l_sum += l.mean().asscalar()        # 使用同步函数
    l.backward()
    trainer.step(X.shape[0])
nd.waitall()
print('increase memory:%f MB' % (get_mem()-mem))

# 异步计算
mem = get_mem()
for X,y in data_iter():
    with autograd.record():
        l = loss(net(X), y)
    l.backward()
    trainer.step(X.shape[0])
nd.waitall()
print('increase memory:%f MB' % (get_mem()-mem))

