import d2lzh as d2l
from mxnet import autograd, nd

# xyplot
def xypolt(x, y, name):
    d2l.set_figsize(figsize=(15,5))
    d2l.plt.figure()
    d2l.plt.plot(x.asnumpy(), y.asnumpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name+'(x)')

# ReLU
x = nd.arange(-8.0, 8.0, 0.1)      # 和matlab差不多
x.attach_grad()
with autograd.record():
    y = x.relu()
y.backward()
xypolt(x, y, 'ReLU')
xypolt(x, x.grad, 'grad of ReLU')


# sigmod
with autograd.record():
    y = x.sigmoid()
y.backward()
xypolt(x, y, 'sigmoid')
xypolt(x, x.grad, 'grad of sigmoid')


# sigmod
with autograd.record():
    y = x.tanh()
y.backward()
xypolt(x, y, 'tanh')
xypolt(x, x.grad, 'grad of tanh')


d2l.plt.show()

