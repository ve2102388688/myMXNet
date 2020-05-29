import d2lzh as d2l
from mxnet.gluon import data as gdata
# import mxnet, numpy                           # 一个简单的ToTensor测试实例 将0-255的数转成0-1的float32
import sys                                      # 系统信息
import time

# 获取数据集 第一次自动下载
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)

# 看看数据集中有多少样本
print('mnist_train:', len(mnist_train), '\tmnist_test:', len(mnist_test))
# 访问一个train sample
feature, label = mnist_train[0]
print('feature:', '\tshape:', feature.shape, '\tdtype:', feature.dtype)        # feature是28*28*1的图像
print('label:', label, '\ttype:', type(label), '\tdtype:', label.dtype)        # label是int32的标签


# 书纸标签转文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# print(get_fashion_mnist_labels(range(2,13)))

# 显示指定范围的图像和标签
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()                                          # Use svg format to display plo
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12,12))    # _表示忽略 不使用该返回值
    for f, img, lbl in zip(figs, images, labels):                   # 从压缩文件中直接读出数据（图像，标签）
        f.imshow(img.reshape((28,28)).asnumpy())
        f.set_title(lbl)                                           # 显示标签
        f.axes.get_xaxis().set_visible(False)                      # 不显示坐标刻度X
        f.axes.get_yaxis().set_visible(False)                      # 不显示坐标刻度Y
    d2l.plt.show()                                                 # Pycharm中没有show()不会出现图片的
# 显示0-9的图像及label
X,y = mnist_train[0:9]
# show_fashion_mnist(X, get_fashion_mnist_labels(y))

# 一个简单的ToTensor测试实例 将0-255的数转成0-1的float32
# transformer = gdata.vision.transforms.ToTensor()
# image = mxnet.nd.random.uniform(0, 255, (4, 2, 3)).astype(dtype=numpy.uint8)
# print(transformer(image))


# 读取FashionMNIST数据集
batch_size = 256
transformer = gdata.vision.transforms.ToTensor()                   # 将0-255的数转成0-1的float32
if sys.platform.startswith('win'):
    num_workers = 0            # windows 不能多线程读取
else:
    num_workers = 4            # Linux 多核读取数据集.奇数不稳定，偶数成倍下降
# transform_first Returns a new dataset with the first element of each sample
train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size, shuffle=True, num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer), batch_size, shuffle=False, num_workers=num_workers)

# 读取所有数据
start = time.time()
for X, y in train_iter:
    continue
print('sec:', (time.time()-start))

