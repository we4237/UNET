
import numpy as np


# input_txt = np.loadtxt('/mnt/user/chenmuyin/depth/UNET_cmy/data/new_masked.txt')
# input_txt = input_txt[input_txt[:,2]<=-0.1]
# x_txt = input_txt[:, 0]
# y_txt = input_txt[:, 1]
# z_txt = input_txt[:, 2] 

# wzq = np.loadtxt('/mnt/user/chenmuyin/depth/UNET_cmy/data/new_wzq.txt')
# wzq_x_min, wzq_x_max = np.min(wzq[:, 0]), np.max(wzq[:, 0])
# wzq_y_min, wzq_y_max = np.min(wzq[:, 1]), np.max(wzq[:, 1])

# points = np.stack([x_txt, y_txt, z_txt], axis=1)
# masked = points[(points[:, 0] >= wzq_x_min) & (points[:, 0] <= wzq_x_max) &
#                 (points[:, 1] >= wzq_y_min) & (points[:, 1] <= wzq_y_max) &
#                 (points[:, 2] <= -0.02) & (points[:, 2] >= -20)]


masked = np.loadtxt('/mnt/user/chenmuyin/depth/UNET_cmy/data/new_masked.txt')
print(len(masked))
# 判断条件，生成布尔索引
condition = np.logical_and(masked[:, 0] <= 562520, masked[:, 1] >= 1.82625e+06)
# 使用布尔索引选择需要保留的数据
new_arr = masked[~condition]

# 判断条件，生成布尔索引
condition = np.logical_and(new_arr[:, 0] <= 562600, new_arr[:, 1] >= 1.8264e+06)
# 使用布尔索引选择需要保留的数据
new_arr = new_arr[~condition]

print(len(new_arr))
np.savetxt('data/new_masked_1.txt', new_arr)