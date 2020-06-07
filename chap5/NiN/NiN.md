################################## lr变化　lr=0.05效果好　###########################################################
# batch_size = 128, lr = 0.5(太大了－失效), resize=224
# sysinfo mem:3325M
epoch 1, loss 2.3000, train acc 0.106, test acc 0.100, time 63.6 sec
epoch 2, loss 2.2937, train acc 0.119, test acc 0.100, time 60.7 sec
epoch 3, loss 2.3026, train acc 0.100, test acc 0.100, time 60.2 sec
epoch 4, loss 2.3026, train acc 0.100, test acc 0.100, time 60.3 sec
epoch 5, loss 2.3026, train acc 0.099, test acc 0.100, time 60.4 sec
# batch_size = 128, lr = 0.1(降低), resize=224
# sysinfo mem:3325M
epoch 1, loss 2.2521, train acc 0.182, test acc 0.100, time 65.0 sec
epoch 2, loss 1.6349, train acc 0.403, test acc 0.614, time 62.8 sec
epoch 3, loss 0.9971, train acc 0.663, test acc 0.678, time 61.7 sec
epoch 4, loss 0.7125, train acc 0.746, test acc 0.782, time 62.3 sec
epoch 5, loss 0.5847, train acc 0.787, test acc 0.779, time 63.3 sec

epoch 1, loss 2.2066, train acc 0.193, test acc 0.218, time 65.1 sec
epoch 2, loss 1.5256, train acc 0.437, test acc 0.610, time 62.1 sec
epoch 3, loss 0.9960, train acc 0.645, test acc 0.677, time 61.9 sec
epoch 4, loss 0.8575, train acc 0.687, test acc 0.722, time 64.1 sec
epoch 5, loss 0.7712, train acc 0.713, test acc 0.728, time 64.1 sec
# batch_size = 128, lr = 0.05(降低), resize=224
# sysinfo mem:3325M
epoch 1, loss 2.2958, train acc 0.160, test acc 0.196, time 64.4 sec
epoch 2, loss 2.2324, train acc 0.192, test acc 0.100, time 61.8 sec
epoch 3, loss 2.2912, train acc 0.139, test acc 0.177, time 61.4 sec
epoch 4, loss 1.8951, train acc 0.328, test acc 0.536, time 61.4 sec
epoch 5, loss 1.1790, train acc 0.616, test acc 0.736, time 62.3 sec
################################## batch_size = 32　###########################################################
# batch_size = 32, lr = 0.05, resize=224
# sysinfo mem:1315M
epoch 1, loss 1.7421, train acc 0.344, test acc 0.721, time 67.9 sec
epoch 2, loss 0.5868, train acc 0.785, test acc 0.854, time 67.4 sec
epoch 3, loss 0.4307, train acc 0.843, test acc 0.867, time 68.0 sec
epoch 4, loss 0.3716, train acc 0.863, test acc 0.877, time 68.6 sec
epoch 5, loss 0.3372, train acc 0.877, test acc 0.898, time 68.1 sec

epoch 1, loss 1.6340, train acc 0.412, test acc 0.733, time 67.8 sec
epoch 2, loss 0.6314, train acc 0.770, test acc 0.811, time 67.5 sec
epoch 3, loss 0.4714, train acc 0.828, test acc 0.828, time 67.6 sec
epoch 4, loss 0.4071, train acc 0.851, test acc 0.876, time 68.2 sec
epoch 5, loss 0.3692, train acc 0.865, test acc 0.820, time 68.2 sec

epoch 1, loss 1.8016, train acc 0.326, test acc 0.573, time 69.2 sec
epoch 2, loss 0.7665, train acc 0.732, test acc 0.827, time 68.4 sec
epoch 3, loss 0.4689, train acc 0.828, test acc 0.833, time 68.8 sec
epoch 4, loss 0.4016, train acc 0.852, test acc 0.871, time 68.6 sec
epoch 5, loss 0.3631, train acc 0.867, test acc 0.878, time 69.4 sec
# batch_size = 32, lr* = 0.06, resize=224
# sysinfo mem:1315M
epoch 1, loss 1.6496, train acc 0.388, test acc 0.686, time 67.9 sec
epoch 2, loss 0.7531, train acc 0.727, test acc 0.824, time 67.4 sec
epoch 3, loss 0.4649, train acc 0.830, test acc 0.840, time 67.4 sec
epoch 4, loss 0.3932, train acc 0.857, test acc 0.872, time 67.5 sec
epoch 5, loss 0.3594, train acc 0.867, test acc 0.865, time 67.6 sec
# batch_size = 32, lr* = 0.04, resize=224
# sysinfo mem:1315M
epoch 1, loss 2.0882, train acc 0.236, test acc 0.201, time 68.4 sec
epoch 2, loss 1.7207, train acc 0.398, test acc 0.537, time 67.5 sec
epoch 3, loss 1.2529, train acc 0.550, test acc 0.582, time 67.3 sec
epoch 4, loss 1.1635, train acc 0.582, test acc 0.593, time 67.4 sec
epoch 5, loss 1.1051, train acc 0.599, test acc 0.601, time 67.8 sec

################################## channels=384变化　###########################################################
256
epoch 1, loss 1.4947, train acc 0.424, test acc 0.716, time 64.8 sec
epoch 2, loss 0.5830, train acc 0.782, test acc 0.801, time 64.2 sec
epoch 3, loss 0.4525, train acc 0.833, test acc 0.862, time 65.2 sec
epoch 4, loss 0.3894, train acc 0.857, test acc 0.883, time 65.1 sec
epoch 5, loss 0.3606, train acc 0.869, test acc 0.885, time 64.6 sec
64
epoch 1, loss 1.6645, train acc 0.375, test acc 0.704, time 59.7 sec
epoch 2, loss 0.6907, train acc 0.738, test acc 0.791, time 59.2 sec
epoch 3, loss 0.5364, train acc 0.800, test acc 0.829, time 59.6 sec
epoch 4, loss 0.4666, train acc 0.828, test acc 0.850, time 59.7 sec
epoch 5, loss 0.4193, train acc 0.845, test acc 0.871, time 59.9 sec
600
epoch 1, loss 1.4942, train acc 0.458, test acc 0.690, time 78.1 sec
epoch 2, loss 0.7859, train acc 0.712, test acc 0.751, time 78.4 sec
epoch 3, loss 0.6718, train acc 0.749, test acc 0.771, time 78.5 sec
epoch 4, loss 0.6144, train acc 0.769, test acc 0.787, time 78.1 sec
epoch 5, loss 0.5822, train acc 0.778, test acc 0.784, time 78.4 sec

################################## nn.Dropout变化　###########################################################
nn.Dropout(0.4)
epoch 1, loss 1.6943, train acc 0.369, test acc 0.732, time 67.5 sec
epoch 2, loss 0.6022, train acc 0.781, test acc 0.844, time 68.1 sec
epoch 3, loss 0.4446, train acc 0.837, test acc 0.858, time 68.0 sec
epoch 4, loss 0.3854, train acc 0.859, test acc 0.876, time 68.6 sec
epoch 5, loss 0.3457, train acc 0.873, test acc 0.893, time 68.6 sec
nn.Dropout(0.3)
epoch 1, loss 1.7376, train acc 0.355, test acc 0.659, time 68.8 sec
epoch 2, loss 0.8429, train acc 0.690, test acc 0.730, time 67.9 sec
epoch 3, loss 0.7082, train acc 0.734, test acc 0.753, time 68.5 sec
epoch 4, loss 0.6386, train acc 0.758, test acc 0.775, time 68.9 sec
epoch 5, loss 0.5958, train acc 0.771, test acc 0.788, time 68.7 sec
nn.Dropout(0.38)
epoch 1, loss 1.5584, train acc 0.438, test acc 0.759, time 67.5 sec
epoch 2, loss 0.5812, train acc 0.790, test acc 0.836, time 68.6 sec
epoch 3, loss 0.4519, train acc 0.836, test acc 0.862, time 68.6 sec
epoch 4, loss 0.3933, train acc 0.856, test acc 0.872, time 67.3 sec
epoch 5, loss 0.3537, train acc 0.870, test acc 0.882, time 67.0 sec
nn.Dropout(0.7)
epoch 1, loss 1.6666, train acc 0.394, test acc 0.734, time 69.8 sec
epoch 2, loss 0.6021, train acc 0.779, test acc 0.819, time 68.3 sec
epoch 3, loss 0.4598, train acc 0.833, test acc 0.859, time 69.5 sec
epoch 4, loss 0.3951, train acc 0.854, test acc 0.874, time 69.2 sec
epoch 5, loss 0.3619, train acc 0.867, test acc 0.880, time 69.2 sec

# 去掉一个1*1的卷积层
epoch 1, loss 1.0910, train acc 0.606, test acc 0.816, time 52.2 sec
epoch 2, loss 0.4838, train acc 0.823, test acc 0.868, time 51.5 sec
epoch 3, loss 0.4048, train acc 0.853, test acc 0.868, time 51.6 sec
epoch 4, loss 0.3643, train acc 0.867, test acc 0.880, time 51.6 sec
epoch 5, loss 0.3342, train acc 0.878, test acc 0.900, time 52.2 sec

