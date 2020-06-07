1. 去掉nn.Dense(120)层，效果瞬间提升
2. 修改nn.Dense(84)的个数，效果变化也是可观的。比如这里的改成60，效果就很不错
3. 学习率可以适当的大写，这里lr=2.5是不错的，这个参数调整过程结果会先增加到了某一个值就下降了
4. epochs次数的提高可以将模型发挥到极致，直到上升不明显为止
5. 在这里激活函数选择其他的，瞬间失效，原因未知?不过sigmoid效果很好
6. 这里的init=init.Xavier()和init.Normal(sigma=0.1)效果差不多，方差为0.2效果更好，大于0.2的效果变得很差
7. kernel_size对结果的影响不是太明显
8. batch_size翻一倍速度慢的多，精度会提升，减少一倍速度很快，精度不够(抖得厉害)，需要折中
9. 建议调节顺序，先看epochs,学习率；调整输出层的数量(尤其靠近输出的那层)，简化模型层数；最后考虑模型的激活函数，kernel_size,init初始化方式

num_epochs, lr = 20, 2.5； batch_size = 256； nn.Dense(60, activation='sigmoid')
epoch 1, loss 1.360741, train acc 0.4834, test_acc 0.7581, time 1.608
epoch 2, loss 0.582017, train acc 0.7741, test_acc 0.8252, time 1.583
epoch 3, loss 0.459555, train acc 0.8282, test_acc 0.8378, time 1.526
epoch 4, loss 0.414496, train acc 0.8451, test_acc 0.8580, time 1.465
epoch 5, loss 0.381215, train acc 0.8595, test_acc 0.8700, time 1.456