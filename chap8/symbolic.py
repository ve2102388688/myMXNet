def add_str():
    return '''
def add(a, b):
    return a+b
    '''

def fancy_func_str():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
    '''

def evoke_str():
    return add_str() + fancy_func_str() + '''
print(fancy_func(1, 2, 3, 4))
    '''

prog = evoke_str()
print(prog)
y = compile(prog, ' ', 'exec')
exec(y)



from mxnet import nd, sym
from mxnet.gluon import nn
import time

def get_net():
    net = nn.HybridSequential()     # 创建HybridSequential实例
    net.add(nn.Dense(256, activation='relu'), 
            nn.Dense(128, activation='relu'), 
            nn.Dense(2))
    net.initialize()
    return net

def benchmark(net, x):
    startTime = time.time()
    for _ in range(1000):
        _ = net(x)
    nd.waitall()                # 等待所有工作完成
    return time.time()-startTime

# Test
x = nd.random.normal(shape=(1, 512))

net = get_net()
print('before: %f' % (benchmark(net, x)))
net.hybridize()     # 启动符号式编程
print('after: %f' % (benchmark(net, x)))

net.export('my_mlp')  # 保存符号式程序.json　模型参数.params

# 输入符号，输出也是符号
x = sym.var('data')
print(net(x))



class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden = nn.Dense(10)
        self.output = nn.Dense(2)
    
    def hybrid_forward(self, F, x):
        # x.asnumpy()     # 不能使用，符号编程中没有实现asnumpy，只能在NDarray下使用
        print('F: ', F)
        print('x: ', x)
        x = F.relu(self.hidden(x))
        # for _ in range(2):
        if 0 == 0:
            print('hidden:', x)
        return self.output(x)

x = nd.random.normal(shape=(1, 4))

net = HybridNet()
net.initialize()
print(net(x))

net.hybridize()
print(net(x))








