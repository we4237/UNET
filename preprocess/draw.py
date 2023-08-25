import random
import laspy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import torch
from osgeo import osr, gdal

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# 设置随机种子
set_seed()
device = torch.device("cuda:0")

# # 读取遥感图像
# input = '/mnt/user/chenmuyin/depth/UNET_cmy/data/cropped/removed'
# # 读取遥感数据
# ds_sample = gdal.Open(input)  # type: gdal.Datasetr
# sample_geotrans = ds_sample.GetGeoTransform()
# sample_proj = ds_sample.GetProjection()
# sample_width = ds_sample.RasterXSize
# sample_height = ds_sample.RasterYSize
# sample_bands = ds_sample.RasterCount
# sample_data = ds_sample.ReadAsArray(0, 0, sample_width, sample_height)
# im_data = np.nan_to_num(sample_data)
# # 从im_data中提取蓝绿红通道数据
# blue_channel = im_data[0]
# green_channel = im_data[1]
# red_channel = im_data[2]
# rgb_image = np.dstack((red_channel, green_channel, blue_channel))
# # 绘制RGB图像
# plt.imshow(rgb_image)
# plt.axis('off')  # 可选，关闭坐标轴
# plt.savefig('1.jpg')

# 读取LAS文件
# las_file = laspy.read(r"/mnt/user/chenmuyin/depth/UNET_cmy/data/pt000001.las")
# 读取抽稀后的LAS
# input = np.loadtxt('/mnt/user/chenmuyin/depth/UNET_cmy/data/downsampled_data.txt')
# x_txt = input[:, 0]
# y_txt = input[:, 1]
# z_txt = input[:, 2]
# z_txt = np.abs(z_txt) 
# z_txt[z_txt > 17] = 17.1
# # 读取npy
# # input = np.load('data/batches_32.npy')
# # input= input[2].reshape((1024, 7))

# # 绘制水深图
# cmap = cm.get_cmap('jet')
# cmap = cmap.reversed()
# plt.scatter(x_txt, y_txt, c=z_txt, s=1, cmap=cmap)
# # plt.imshow(x_txt, y_txt, c=z_txt, s=1, cmap='jet', vmin=0, vmax=17.1)  # 使用 'gray' 颜色映射，可根据需求更改
# plt.colorbar()  # 添加颜色条
# plt.axis('off')
# plt.show()
# plt.savefig('output/liadr.jpg', bbox_inches='tight')



# 读取遥感图像
input_file = '/mnt/user/chenmuyin/depth/UNET_cmy/data/cropped/masked.dat'
ds = gdal.Open(input_file)
sample_geotrans = ds.GetGeoTransform()
sample_proj = ds.GetProjection()
# 获取图像宽度和高度
width = ds.RasterXSize
height = ds.RasterYSize


# 计算像素坐标转换为经纬度坐标的转换比例
x_ratio = (sample_geotrans[1] * width) / width
y_ratio = (sample_geotrans[5] * height) / height

# 计算左上角的经纬度坐标
top_left_x = sample_geotrans[0]
top_left_y = sample_geotrans[3]

# 计算右下角的经纬度坐标
bottom_right_x = top_left_x + (width * x_ratio)
bottom_right_y = top_left_y + (height * y_ratio)
x_ticks = range(0, width, 5000)  # 设置刻度位置
x_ticks = x_ticks[1:]
y_ticks = range(0, height, 5000)  # 设置刻度位置
y_ticks = y_ticks[1:]
# 生成横纵坐标轴的经度和纬度值
x_coords = np.linspace(top_left_x, bottom_right_x, width // 5000)
y_coords = np.linspace(top_left_y, bottom_right_y, height // 5000)
x_list = [[x,0] for x in x_coords]
y_list = [[0,y] for y in y_coords]

# 投影转经纬度
prosrs = osr.SpatialReference()
prosrs.ImportFromWkt(ds.GetProjection())
geosrs = prosrs.CloneGeogCS()
ct = osr.CoordinateTransformation(prosrs, geosrs)
lon_list = ct.TransformPoints(x_list)
lat_list = ct.TransformPoints(y_list)
lon = np.array(lon_list)[:,0]
lat = np.array(lat_list)[:,1]
x_label = [f'{x:.2f}' for x in lon]
y_label = [f'{y:.2f}' for y in lat]

plt.xticks(x_ticks, x_label, rotation=45)  # 设置刻度位置和标签
# plt.xlabel("经度")  # 设置横轴标题

plt.yticks(y_ticks, y_label)  # 设置刻度位置和标签
# plt.ylabel("纬度")  # 设置纵轴标题

input = np.load('output/depth_1000.npy')

# 反转颜色映射
c = cm.get_cmap('jet')
c = c.reversed()
plt.imshow(input, cmap=c, vmin=0, vmax=17.1)  # 使用 'gray' 颜色映射，可根据需求更改
plt.colorbar()  # 添加颜色条
plt.show()
plt.savefig('output/res.jpg', dpi=300, bbox_inches='tight')