import d2lzh as d2l
import mxnet as mx
from mxnet import nd

################################################# TODO 并行计算 #####################################################
# 定义一个测试函数
def run(x):
    return [nd.dot(x, x) for _ in range(10)]

# 分别在GPU,CPU上创建数据
x_cpu = nd.random.uniform(shape=(2000,2000))
x_gpu = nd.random.uniform(shape=(2000,2000), ctx=mx.gpu())

# run(x_cpu)        # 预热开始
# run(x_gpu)
# nd.waitall()      # 预热结束

with d2l.Benchmark('Run on CPU.'):          # 0.7571 sec
    run(x_cpu)
    nd.waitall()
with d2l.Benchmark('Run on GPU.'):          # 0.0515 sec
    run(x_gpu)
    nd.waitall()

# 时间小于分开的和
with d2l.Benchmark('Both on CPU, GPU.'):    # 0.7544 sec
    run(x_cpu)
    run(x_gpu)
    nd.waitall()


################################################# TODO CPUh和GPU的通信 #####################################################
def copy_to_cpu(x):
    return [y.copyto(mx.cpu()) for y in x]

# 串行
with d2l.Benchmark('Run on GPU.'):      # 0.0494 sec
    y = run(x_gpu)
    nd.waitall()
with d2l.Benchmark('Copy to CPU.'):     # 0.1153 sec
    copy_to_cpu(y)
    nd.waitall()

# 并行
with d2l.Benchmark('Run and copy.'):    # 0.1082 sec
    y = run(x_gpu)
    copy_to_cpu(y)
    nd.waitall()

