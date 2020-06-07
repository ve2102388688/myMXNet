# 没有添加批量归一化
num_epochs, lr = 20, 2.5； batch_size = 256； nn.Dense(60, activation='sigmoid')
epoch 1, loss 1.360741, train acc 0.4834, test_acc 0.7581, time 1.608
epoch 2, loss 0.582017, train acc 0.7741, test_acc 0.8252, time 1.583
epoch 3, loss 0.459555, train acc 0.8282, test_acc 0.8378, time 1.526
epoch 4, loss 0.414496, train acc 0.8451, test_acc 0.8580, time 1.465
epoch 5, loss 0.381215, train acc 0.8595, test_acc 0.8700, time 1.456

# 添加批量归一化
# 此时lr可以取很大的范围，20都没有问题
epoch 1, loss 0.6568, train acc 0.764, test acc 0.863, time 2.8 sec
epoch 2, loss 0.3964, train acc 0.857, test acc 0.868, time 2.7 sec
epoch 3, loss 0.3469, train acc 0.874, test acc 0.881, time 2.7 sec
epoch 4, loss 0.3198, train acc 0.884, test acc 0.890, time 2.7 sec
epoch 5, loss 0.2983, train acc 0.893, test acc 0.893, time 2.6 sec
# gluon要快一点	　
epoch 1, loss 0.8502, train acc 0.740, test acc 0.836, time 1.9 sec
epoch 2, loss 0.4013, train acc 0.852, test acc 0.829, time 1.8 sec
epoch 3, loss 0.3498, train acc 0.871, test acc 0.836, time 1.7 sec
epoch 4, loss 0.3199, train acc 0.882, test acc 0.855, time 1.8 sec
epoch 5, loss 0.2990, train acc 0.891, test acc 0.878, time 1.8 sec
# use_global_stats=True 全局性批量归一化，效果不是太好
# scale=False 如果为True，则乘以gamma。 如果为False，则不使用gamma。当下一层是线性的（例如nn.relu）时，可以禁用此功能，因为缩放将由下一层完成。





















