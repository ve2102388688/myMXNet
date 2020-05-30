from mxnet import nd
from mxnet.gluon import nn

# MLP继承nn.Block
class MLP(nn.Block):                    # Block是python2.7写的
    def __init__(self, **kwargs):
        # 初始化父类的属性
        super().__init__(**kwargs)            # 'MLP' object has no attribute '_children'
        self.hidden = nn.Dense(256, activation='relu')  # 隐藏层
        self.output = nn.Dense(10)                      # 输出层

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        return self.output(self.hidden(x))

X = nd.random.uniform(shape=(2,20))
net = MLP()
net.initialize()
print(net(X))



class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add(self, block):
        """ block是一个Block实例　保存在Block的_children, 调用initialize会自动初始化_children所有成员　"""
        self._children[block.name] = block

    def forward(self, x):
        """ OrderedDict保证按添加顺序成员成员　"""
        for block in self._children.values():   # values是一个迭代器
            x = block(X)                        # forward() missing 1 required positional argument: 'x'
        return x

net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
print(net(X))


class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # get_constant创建随机权重在训练中不会被迭代
        self.rand_weight = self.params.get_constant('rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        x = nd.relu(nd.dot(x, self.rand_weight.data())+1)   # 创建常数参数
        x = self.dense(x)                                   # 复用全连接层　等价于两个全连接层共享参数
        while x.norm() > 1:
            x /= 2
        if x.norm() < 0.8:
            x *= 10
        return x.sum()

net = FancyMLP()
net.initialize()
print(net(X))


class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential();
        self.net.add(nn.Dense(64, activation='relu'), nn.Dense(32, activation='relu'))
        # self.register_child(self.net)
        # self.net = [nn.Dense(64, activation='relu'), nn.Dense(32, activation='relu')]
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

net = nn.Sequential()
net.add(NestMLP(), nn.Dense(20), FancyMLP())
net.initialize()
print(net(X))


