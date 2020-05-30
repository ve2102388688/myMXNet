from mxnet import nd
from mxnet.gluon import nn


"""**************************** max/avg pooling ****************************************"""
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    output = nd.zeros((X.shape[0]-p_h+1, X.shape[1]-p_w+1))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if mode == 'max':
                output[i, j] = X[i:(i+p_h), j:(j+p_w)].max()
            elif mode == 'avg':
                output[i, j] = X[i:(i+p_h), j:(j+p_w)].mean()

    return output

X = nd.array([[0,1,2], [3,4,5], [6,7,8]])
print(pool2d(X, (2,2)))
print(pool2d(X, (2,2), 'avg'))



"""**************************** padding/stride ****************************************"""
X = nd.arange(16).reshape((1,1,4,4))
print('Orange:', X)

# Attention 默认情况下,strides=K_h(K_w)
mypool2d = nn.MaxPool2D(3)              # equals to nn.MaxPool2D(3, strides=2)
print('\nDefault:', mypool2d(X))

mypool2d = nn.MaxPool2D(3, padding=1, strides=2)
print('\npadding=1, strides=2',mypool2d(X))

mypool2d = nn.MaxPool2D((2,3), padding=(1,2), strides=(2,3))
print('\nnn.MaxPool2D((2,3), padding=(1,2), strides=(2,3))',mypool2d(X))



"""**************************** Muti chaneels ****************************************"""
X = nd.concat(X, X+1, dim=1)
print(X)
mypool2d = nn.MaxPool2D(3, padding=1, strides=2)
print('\nMuti chaneels,padding=1, strides=2',mypool2d(X))
