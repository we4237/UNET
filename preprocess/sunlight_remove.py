import sys
from statistics import mean
from typing import List
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from osgeo import gdal, gdalconst
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

def calculate_Coef(NIR,Band)->float:
    #传参
    X = NIR
    Y = Band
    # 模型搭建
    model = LinearRegression()
    model.fit(X, Y)
    # 模型可视化
    # plt.scatter(X, Y)
    # plt.plot(X, model.predict(X), color='red')
    # plt.xlabel('NIR')
    # plt.ylabel('Band')
    # plt.text(600, 600, 'R^2=' + str(np.round(r2_score(X,Y), 3)), family='Times New Roman')
    # plt.text(600, 800, 'Coef' + str(np.round(model.coef_, 3)), family='Times New Roman')
    # plt.show()
    # 查看回归系数
    b = float(model.coef_[0])
    print('回归系数：'+ str(("%.3f" % b)))
    return model.coef_

def calculate_Min(NIR)->float:
    return min(NIR)

def sunlight_Correction(R_i,R_NIR,coef,min)->List[float]:
    return R_i-coef*(R_NIR-min)

# 计算参数
input = '/mnt/user/chenmuyin/depth/UNET_cmy/data/cropped/subset.dat'
ds_sample = gdal.Open(input)  # type: gdal.Datasetr
sample_geotrans = ds_sample.GetGeoTransform()
sample_proj = ds_sample.GetProjection()
sample_width = ds_sample.RasterXSize
sample_height = ds_sample.RasterYSize
sample_bands = ds_sample.RasterCount
sample_data = ds_sample.ReadAsArray(0, 0, sample_width, sample_height)
print(sample_proj)
print(sample_geotrans)
print('数据的大小（行，列）：')
print('(%s %s)' % (ds_sample.RasterYSize, ds_sample.RasterXSize))

sample = sample_height * sample_width
arr_sample = [[0] * sample_bands for _ in range(0, sample)]
for i in range(sample_bands):
     arr_sample[i] = np.ravel(sample_data[i, 0:sample_height, 0:sample_width]).reshape(sample, 1)

# 过程绘图
# blue_=sunlight_Correction(arr_sample[0],arr_sample[3],calculate_Coef(),calculate_Min(arr_sample[3])).reshape(sample_height, sample_width)
# green_=sunlight_Correction(arr_sample[1],arr_sample[3],calculate_Coef(),calculate_Min(arr_sample[3])).reshape(sample_height, sample_width)
# red_=sunlight_Correction(arr_sample[2],arr_sample[3],calculate_Coef(),calculate_Min(arr_sample[3])).reshape(sample_height, sample_width)


# # 传参
Blue_coef = calculate_Coef(arr_sample[3],arr_sample[0])
Green_coef = calculate_Coef(arr_sample[3],arr_sample[1])
Red_coef = calculate_Coef(arr_sample[3],arr_sample[2])
NIR_coef = calculate_Coef(arr_sample[3],arr_sample[3])
# b5_coef = calculate_Coef(arr_sample[3],arr_sample[4])
# b6_coef = calculate_Coef(arr_sample[3],arr_sample[5])
# b7_coef = calculate_Coef(arr_sample[3],arr_sample[6])
# b8_coef = calculate_Coef(arr_sample[3],arr_sample[7])
NIR_min = calculate_Min(arr_sample[3])

# coef = np.loadtxt('output/coef.txt')
# Blue_coef = coef[0]
# Green_coef = coef[1]
# Red_coef = coef[2]
# NIR_coef = coef[3]
# NIR_min = coef[4]

# # 耀斑去除
inputname = '/mnt/user/chenmuyin/depth/UNET_cmy/data/cropped/masked.dat'
ds_result = gdal.Open(inputname)  # type: gdal.Datasetr
result_geotrans = list(ds_result.GetGeoTransform())
result_proj = ds_result.GetProjection()
result_width = ds_result.RasterXSize
result_height = ds_result.RasterYSize
result_bands = ds_result.RasterCount
result_data = ds_result.ReadAsArray(0, 0, result_width, result_height)
print('数据的大小（行，列）：')
print('(%s %s)' % (ds_result.RasterYSize, ds_result.RasterXSize))


large = result_height * result_width
arr_result = [[0] * result_bands for _ in range(0, large)]
for i in range(4):
     arr_result[i] = np.ravel(result_data[i, 0:result_height, 0:result_width]).reshape(large, 1)
blue_=sunlight_Correction(arr_result[0],arr_result[3],
                          Blue_coef,NIR_min).reshape(result_height, result_width)
green_=sunlight_Correction(arr_result[1],arr_result[3],
                           Green_coef,NIR_min).reshape(result_height, result_width)
red_=sunlight_Correction(arr_result[2],arr_result[3],
                         Red_coef,NIR_min).reshape(result_height, result_width)
NIR_=sunlight_Correction(arr_result[3],arr_result[3],
                         NIR_coef,NIR_min).reshape(result_height, result_width)
# b5_=sunlight_Correction(arr_result[4],arr_result[3],
#                          b5_coef,NIR_min).reshape(result_height, result_width)
# b6_=sunlight_Correction(arr_result[5],arr_result[3],
#                          b6_coef,NIR_min).reshape(result_height, result_width)
# b7_=sunlight_Correction(arr_result[6],arr_result[3],
#                          b7_coef,NIR_min).reshape(result_height, result_width)
# b8_=sunlight_Correction(arr_result[7],arr_result[3],
#                          b8_coef,NIR_min).reshape(result_height, result_width)

blue_ ,green_,red_,NIR_ = np.array(blue_),np.array(green_),np.array(red_),np.array(NIR_)
data=np.array([blue_,green_,red_,NIR_])

# 输出结果
resultname='/mnt/user/chenmuyin/depth/UNET_cmy/data/cropped/removed'
# driver = gdal.GetDriverByName("ENVI")
dataset = gdal.Open(inputname, gdalconst.GA_ReadOnly)
band1 = dataset.GetRasterBand(1)
data_type = band1.DataType
target = dataset.GetDriver().Create(resultname, result_width, result_height, 4, data_type)
# geoname = r'D:\SHOU\NSG(0717)_04.16\GF6cut.dat'
# ds_ = gdal.Open(geoname)
# geotrans = ds_.GetGeoTransform()
# prog = ds_.GetProjection()
target.SetProjection(result_proj)  # 写入投影
target.SetGeoTransform(result_geotrans)  # 写入仿射变换参数


for i in range(4):
    
    out_band  = target.GetRasterBand(i+1)
    out_band.WriteArray(data[i])
    out_band.FlushCache()


