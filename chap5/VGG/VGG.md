# batch_size = 32, lr = 0.05, resize=224
# sysinfo mem:4427M
epoch 1, loss 0.6547, train acc 0.758, test acc 0.870, time 410.5 sec
epoch 2, loss 0.3370, train acc 0.880, test acc 0.901, time 407.3 sec

epoch 1, loss 0.6622, train acc 0.757, test acc 0.880, time 391.6 sec
epoch 2, loss 0.3352, train acc 0.881, test acc 0.904, time 388.4 sec

epoch 1, loss 0.6563, train acc 0.758, test acc 0.884, time 394.8 sec
epoch 2, loss 0.3361, train acc 0.879, test acc 0.904, time 388.0 sec
epoch 3, loss 0.2748, train acc 0.902, test acc 0.901, time 389.0 sec
epoch 4, loss 0.2398, train acc 0.914, test acc 0.921, time 388.2 sec
epoch 5, loss 0.2072, train acc 0.925, test acc 0.929, time 387.8 sec

epoch 1, loss 0.6657, train acc 0.754, test acc 0.879, time 388.5 sec
epoch 2, loss 0.3403, train acc 0.879, test acc 0.910, time 383.2 sec
epoch 3, loss 0.2771, train acc 0.900, test acc 0.917, time 383.9 sec
epoch 4, loss 0.2374, train acc 0.914, test acc 0.907, time 388.0 sec
epoch 5, loss 0.2107, train acc 0.924, test acc 0.925, time 389.5 sec
epoch 6, loss 0.1864, train acc 0.933, test acc 0.926, time 388.7 sec
epoch 7, loss 0.1651, train acc 0.940, test acc 0.933, time 386.9 sec
epoch 8, loss 0.1458, train acc 0.946, test acc 0.931, time 387.4 sec
epoch 9, loss 0.1307, train acc 0.952, test acc 0.934, time 389.3 sec
epoch 10, loss 0.1169, train acc 0.958, test acc 0.936, time 389.9 sec


# 输入减少，不仅训练时间大大减少。而且占用的显存也减少很多，下面0.05的学习率很好
# batch_size＝32,128显存明显加倍，但是时间减少不多，32效益更好
# nit.Xavier()效果好，但是正太的不能工作nan
# batch_size* = 32, lr = 0.05, resize=96
# sysinfo mem:1475M
epoch 1, loss 0.7608, train acc 0.717, test acc 0.839, time 81.8 sec
epoch 2, loss 0.3771, train acc 0.864, test acc 0.884, time 80.9 sec
epoch 3, loss 0.3055, train acc 0.890, test acc 0.909, time 84.4 sec
epoch 4, loss 0.2622, train acc 0.906, test acc 0.913, time 84.5 sec
epoch 5, loss 0.2309, train acc 0.917, test acc 0.920, time 83.0 sec

epoch 1, loss 0.7365, train acc 0.730, test acc 0.863, time 90.1 sec
epoch 2, loss 0.3342, train acc 0.878, test acc 0.895, time 89.3 sec
epoch 3, loss 0.2725, train acc 0.901, test acc 0.913, time 89.4 sec
epoch 4, loss 0.2353, train acc 0.914, test acc 0.915, time 89.3 sec
epoch 5, loss 0.2081, train acc 0.922, test acc 0.921, time 88.6 sec

# # batch_size = 32, lr = 0.05, resize=96, VGG16
# sysinfo mem:2360M
epoch 1, loss 0.8979, train acc 0.664, test acc 0.856, time 174.4 sec
epoch 2, loss 0.3480, train acc 0.871, test acc 0.892, time 172.9 sec
epoch 3, loss 0.2830, train acc 0.896, test acc 0.896, time 173.4 sec
epoch 4, loss 0.2440, train acc 0.911, test acc 0.920, time 173.1 sec
epoch 5, loss 0.2152, train acc 0.921, test acc 0.914, time 174.4 sec

# # batch_size = 32, lr = 0.05, resize=96, VGG19
# sysinfo mem:2525M
epoch 1, loss 2.3024, train acc 0.103, test acc 0.100, time 199.9 sec
epoch 2, loss 2.3035, train acc 0.099, test acc 0.100, time 196.8 sec
epoch 3, loss 2.2723, train acc 0.119, test acc 0.492, time 198.8 sec
epoch 4, loss 0.6532, train acc 0.757, test acc 0.855, time 198.4 sec
epoch 5, loss 0.3688, train acc 0.864, test acc 0.873, time 198.7 sec
# 可以看出VGG11表现好

# batch_size* = 128, lr = 0.05, resize=96
# sysinfo mem:3541M
epoch 1, loss 1.1664, train acc 0.568, test acc 0.803, time 77.4 sec
epoch 2, loss 0.5378, train acc 0.802, test acc 0.847, time 71.4 sec
epoch 3, loss 0.4308, train acc 0.841, test acc 0.873, time 72.3 sec
epoch 4, loss 0.3752, train acc 0.863, test acc 0.880, time 73.3 sec
epoch 5, loss 0.3364, train acc 0.878, test acc 0.892, time 73.6 sec
# batch_size* = 200, lr = 0.05, resize=96
# sysinfo mem:5135M
epoch 1, loss 1.3985, train acc 0.492, test acc 0.763, time 77.5 sec
epoch 2, loss 0.6305, train acc 0.767, test acc 0.834, time 75.4 sec
epoch 3, loss 0.4884, train acc 0.822, test acc 0.850, time 75.1 sec
epoch 4, loss 0.4264, train acc 0.844, test acc 0.863, time 74.8 sec
epoch 5, loss 0.3827, train acc 0.861, test acc 0.875, time 75.6 sec

epoch 1, loss 1.1576, train acc 0.573, test acc 0.783, time 81.4 sec
epoch 2, loss 0.5018, train acc 0.814, test acc 0.858, time 76.6 sec
epoch 3, loss 0.3943, train acc 0.856, test acc 0.875, time 77.1 sec
epoch 4, loss 0.3459, train acc 0.873, test acc 0.889, time 77.4 sec
epoch 5, loss 0.3094, train acc 0.886, test acc 0.893, time 76.1 sec
# batch_size = 128, lr* = 0.2, resize=96 效果变差
# sysinfo mem:4079M
epoch 1, loss 2.3197, train acc 0.112, test acc 0.100, time 75.5 sec
epoch 2, loss 2.3034, train acc 0.106, test acc 0.100, time 70.2 sec
epoch 3, loss 2.1772, train acc 0.153, test acc 0.474, time 70.7 sec
epoch 4, loss 0.7176, train acc 0.727, test acc 0.828, time 72.9 sec
epoch 5, loss 0.4283, train acc 0.844, test acc 0.868, time 72.5 sec
# batch_size = 128, lr* = 0.１, resize=96 效果变差
# sysinfo mem:3540M
epoch 1, loss 1.0009, train acc 0.625, test acc 0.833, time 75.6 sec
epoch 2, loss 0.4590, train acc 0.832, test acc 0.867, time 72.5 sec
epoch 3, loss 0.3671, train acc 0.865, test acc 0.888, time 71.8 sec
epoch 4, loss 0.3163, train acc 0.885, test acc 0.895, time 73.1 sec
epoch 5, loss 0.2842, train acc 0.897, test acc 0.905, time 72.2 sec
# batch_size = 128, lr* = 0.07, resize=96 效果变差
# sysinfo mem:3540M
epoch 1, loss 1.0827, train acc 0.597, test acc 0.816, time 77.6 sec
epoch 2, loss 0.5008, train acc 0.816, test acc 0.860, time 72.8 sec
epoch 3, loss 0.3990, train acc 0.854, test acc 0.868, time 72.1 sec
epoch 4, loss 0.3484, train acc 0.874, test acc 0.893, time 72.2 sec
epoch 5, loss 0.3137, train acc 0.886, test acc 0.898, time 72.2 sec
# batch_size = 128, lr* = 0.03, resize=96 效果变差
# sysinfo mem:3540M
epoch 1, loss 1.3686, train acc 0.503, test acc 0.746, time 76.8 sec
epoch 2, loss 0.6294, train acc 0.767, test acc 0.822, time 73.1 sec
epoch 3, loss 0.4938, train acc 0.819, test acc 0.851, time 73.2 sec
epoch 4, loss 0.4329, train acc 0.842, test acc 0.866, time 73.1 sec
epoch 5, loss 0.3923, train acc 0.857, test acc 0.875, time 73.0 sec

