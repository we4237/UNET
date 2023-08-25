from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

# input_txt = np.loadtxt('/mnt/user/chenmuyin/depth/UNET_cmy/data/wzq.txt')
# # 获取XYZ坐标和水深信息
# input_txt = input_txt[input_txt[:, 2] < 0]
# x_txt = input_txt[:, 0]
# y_txt = input_txt[:, 1]
# x_txt_min, x_txt_max = np.min(x_txt), np.max(x_txt)
# y_txt_min, y_txt_max = np.min(y_txt), np.max(y_txt)

# 上采样
# # 读取原始数据
# data = np.loadtxt('/mnt/user/chenmuyin/depth/UNET_cmy/data/new_masked_1.txt')
# x = data[:, 0]
# y = data[:, 1]
# z = data[:, 2] 

# # 设置新的采样点间距
# new_spacing = 10

# # 计算新采样点的范围
# x_min, x_max = np.min(x), np.max(x)
# y_min, y_max = np.min(y), np.max(y)

# # 计算新采样点的个数
# nx = int(np.ceil((x_max - x_min) / new_spacing)) #1087
# ny = int(np.ceil((y_max - y_min) / new_spacing)) #2052

# # 生成新采样点的网格
# new_x, new_y = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))

# # 使用 interpolate.griddata 进行重采样
# new_z = interpolate.griddata((x, y), z, (new_x, new_y), method='linear',rescale=True)


# # 保存结果
# np.savetxt('data/resample_new_liner.txt', np.column_stack((new_x.ravel(), new_y.ravel(), new_z.ravel())))



# 下采样

# 读取数据文件
df = pd.read_csv('/mnt/user/chenmuyin/depth/UNET_cmy/data/new_masked_1.txt', sep=' ', names=['x', 'y', 'z'])

# 将浮点数坐标转换为整数坐标
df['x_int'] = (df['x'] * 10).astype(int)
df['y_int'] = (df['y'] * 10).astype(int)

# 计算每个区域的中心点坐标
x_min = df['x_int'].min()
x_max = df['x_int'].max()
y_min = df['y_int'].min()
y_max = df['y_int'].max()
x_bins = np.arange(x_min, x_max + 1, 10)
y_bins = np.arange(y_min, y_max + 1, 10)
df['x_center'] = df['x_int'].apply(lambda x: x_bins[np.argmin(np.abs(x_bins - x))] + 5)
df['y_center'] = df['y_int'].apply(lambda y: y_bins[np.argmin(np.abs(y_bins - y))] + 5)

# 将数据按照中心点坐标分组，并取每组中距离中心点最近的数据点
grouped = df.groupby(['x_center', 'y_center'])
result = grouped.apply(lambda x: x.iloc[((x['x'] - x.name[0])**2 + (x['y'] - x.name[1])**2).argmin()])

# 保存结果数据
result[['x', 'y', 'z']] .to_csv("data/downsampled_data.txt", header=None, index=None, sep=' ', float_format='%.2f')
