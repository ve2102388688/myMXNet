from mxnet import nd
import random
import zipfile


X, W_xh = nd.random.uniform(shape=(3, 1)), nd.random.uniform(shape=(1, 4))
H, W_hh = nd.random.uniform(shape=(3, 4)), nd.random.uniform(shape=(4, 4))

print(nd.dot(X, W_xh) + nd.dot(H, W_hh))
print(nd.dot(nd.concat(X, H, dim=1), nd.concat(W_xh, W_hh, dim=0)))


# 读取歌词文件,6w字
with zipfile.ZipFile('aychou_lyrics.txt.zip') as zin: # 打开压缩文件jaychou_lyrics.txt.zip
    with zin.open('jaychou_lyrics.txt') as fd:                              # 获取里面的一个叫jaychou_lyrics.txt的文件
        corpus_chars = fd.read().decode('utf-8')                            # utf-8解码
# print(corpus_chars[:40])    # 展示前40个字

# 用前1w字训练模型
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:10000]       #print(corpus_chars)



# 建立字符索引, 索引到字
idx_to_char = list(set(corpus_chars))   # set创建无序不重复序列并转成列表,这里共有1027个不同的字
# 创建字典(char<->idx), 字到索引
char_to_idx = dict([(char, i) for i,char in enumerate(idx_to_char)])    # 列表解析,要求[]括起来, i,char不能换
print(len(char_to_idx)) # 1027

# 歌词的对应的index
corpus_indices = [char_to_idx[char] for char in corpus_chars]   # 根据char在字典中找到idx,前１w个字
sample = corpus_indices[:20]        # 看看前面20个字
print('char: ', ''.join([idx_to_char[idx] for idx in sample]))  # idx->char,'A'.join(B):B中字符用A间隔
print('indices: ', sample)



# 随机采样
# num_step:序列长度
def data_iter_random(corpus_indices, batch_size, num_step, ctx=None):
    num_example = (len(corpus_indices)-1) // num_step           # 一堆数据分成每块num_step个
    num_epochs = num_example // batch_size                      # 取batch_size个num_step成一包
    # 取num_example个数并打乱, 标号1,2,,,num_example,下面就是每次在里面随机拿batch_size个
    # 比如30个数据按6个一组,共有5组,记为0,1,2,3,4;打乱,每次随机取batch_size个,也就是2个,当然只能取2次,
    # 如果取0,1,对应着0-6,7-11
    example_indices = list(range(num_example))
    random.shuffle(example_indices)

    # 返回从pos开始num_step序列
    def _data(pos):
        return corpus_indices[pos:pos+num_step]

    # 明显corpus_indices//num_step//batch_size次循环
    for i in range(num_epochs):
        i = i * batch_size
        batch_indices = example_indices[i:i+batch_size]     # 获得标号集,由于打乱了,每次都不一样
        X = [_data(j*num_step) for j in batch_indices]      # 开始取数据
        Y = [_data(j*num_step+1) for j in batch_indices]    # 无太大意义,完全是上面的移位1
        yield nd.array(X, ctx), nd.array(Y, ctx)

# Test
my_seq = list(range(30))
print('\ndata_iter_random:')
for X,Y in data_iter_random(my_seq, batch_size=2, num_step=6):
    print('X: ', X, '\nY: ', Y, '\n')


# 相邻采样
# num_step:序列长度
def data_iter_consecutive(corpus_indices, batch_size, num_step, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx)
    batch_len = len(corpus_indices) // batch_size   # 一块有多少数据
    indices = corpus_indices[0:batch_len*batch_size].reshape((batch_size, batch_len))   # 矩阵(batch_size, batch_len),多的舍去

    num_epochs = batch_len // num_step          # corpus_indices//num_step//batch_size次循环
    for i in range(num_epochs):
        i = i * num_step                        # 每次拿6个
        X = indices[:, i:i+num_step]            # 15个数取6个
        Y = indices[:, i+1:i+1+num_step]        # 无太大意义,完全是上面的移位1
        yield X, Y

# Test
print('\ndata_iter_consecutive:')
for X,Y in data_iter_consecutive(my_seq, batch_size=2, num_step=6):
    print('X: ', X, '\nY: ', Y, '\n')











