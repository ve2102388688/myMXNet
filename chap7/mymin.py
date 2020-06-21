import d2lzh as d2l
from mpl_toolkits import mplot3d
import numpy as np

# TODO f(x)=xcos(pi*x) [-1.0, 2.0]
def f(x):
    return x*np.cos(np.pi * x)

d2l.set_figsize(figsize=(15,5))
x = np.arange(-1.0, 2.0, 0.1)
fig, = d2l.plt.plot(x, f(x))
fig.axes.annotate('local minimum', xy=(-0.3, -0.25), xytext=(-0.77, -1.0),
                  arrowprops=dict(arrowstyle='->'))
fig.axes.annotate('global minimum', xy=(1.1, -0.95), xytext=(0.6, 0.8),
                  arrowprops=dict(arrowstyle='->'))
d2l.plt.xlabel('x')
d2l.plt.ylabel('f(x)')


# TODO f(x)=x^3 [-2, 2.0]
x = np.arange(-2.0, 2.0, 0.1)
fig, = d2l.plt.plot(x, x**3)
# 给曲线添加一些注释,xy是标记点,xytext是文字的位置
fig.axes.annotate('saddle point', xy=(0,0), xytext=(-0.52,-5.0), arrowprops=dict(arrowstyle='->'))
# 坐标轴
d2l.plt.xlabel('x')
d2l.plt.ylabel('f(x)')


# TODO f(x)=x^2-y^2 [-2, 2.0]
x, y = np.mgrid[-1:1:31j, -1:1:31j]
z = x**2 - y**2
ax = d2l.plt.figure().add_subplot(111, projection='3d')

ax.plot_wireframe(x, y, z, **{'rstride':2, 'cstride':2})    # **{'rstride':2, 'cstride':2}　图像不要那么密
# 红色标记鞍点
ax.plot([0], [0], [0], 'rx')
# 三个方向上刻度只有　-1, 0, 1
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks) 
d2l.plt.yticks(ticks) 
ax.set_zticks(ticks)
# 坐标轴
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')


d2l.plt.show()


