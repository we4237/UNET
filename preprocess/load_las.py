
import laspy
import numpy as np
import matplotlib.pyplot as plt

# # 读取LAS文件
# las_file = laspy.read("data/pt000001.las")

# # 获取XYZ坐标和水深信息
# x = las_file.x
# y = las_file.y
# z = las_file.z

# 读取抽稀后的txt
input_txt = np.loadtxt('/mnt/user/chenmuyin/depth/UNET_cmy/data/downsampled_data.txt')

# 获取XYZ坐标和水深信息
x = input_txt[:, 0]
y= input_txt[:, 1]
z= input_txt[:, 2] 

# 输出长宽高以及空间分辨率
x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)
z_min, z_max = np.min(z), np.max(z)
dx, dy, dz = np.mean(np.diff(np.unique(x))), np.mean(np.diff(np.unique(y))), np.mean(np.diff(np.unique(z)))

print("长:", x_max - x_min, "宽:", y_max - y_min, "高:", z_max - z_min, "空间分辨率(dx, dy, dz):", dx, dy, dz)


