import random
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from osgeo import osr, gdal
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,median_absolute_error,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from sklearn.pipeline import make_pipeline
from scipy import stats
from statistics import mean

import torch
from Unet import UNET

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def scatter_out(x, y, title_name, savename):  ## x,y为两个需要做对比分析的两个量。
    # x = y_true
    # y = y_pred
    # ==========计算评价指标==========
    BIAS = cv2.mean(x - y)
    MSE = mean_squared_error(x, y)
    RMSE = np.power(MSE, 0.5)
    R2 = r2_score(x, y)
    MAE = mean_absolute_error(x, y)
    MRE = mean_absolute_percentage_error(x, y)

    # ===========Calculate the point density==========
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)
    # ===========Sort the points by density, so that the densest points are plotted last===========
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    z = np.linspace(0,1,len(x))

    def best_fit_slope_and_intercept(xs, ys):
        m = (((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) * mean(xs)) - mean(xs * xs)))
        b = mean(ys) - m * mean(xs)
        return m, b

    m, b = best_fit_slope_and_intercept(x, y)

    regression_line = []
    for a in x:
        regression_line.append((m * a) + b)

    plt.plot([0, 18], [0, 18], 'black', lw=0.8)  # 画的1:1线，线的颜色为black，线宽为0.8
    plt.scatter(x, y, c=z, s=7, cmap='jet')

    # edgecolor: 设置轮廓颜色的,none是无色
    # 将c设置成一个y值列表并使用参数 cmap.cm.XX来使用某个颜色渐变(映射)
    # https://matplotlib.org/examples/color/colormaps_reference.html 从这个网页可以获得colorbard的名称
    # c=是设置数据点颜色的,在2.0的matplotlib中,edgecolor默认为'none'
    # c=还可以使用RGB颜色,例如 c = (0,0,0.8) 分别表示 红绿蓝,值越接近0,颜色越深.越接近1,颜色越浅

    plt.plot(x, regression_line, 'red', lw=1)  # 预测与实测数据之间的回归线
    plt.axis([0, 18, 0, 18])  # 设置线的范围

    plt.xlabel("in-situ", family='Times New Roman')
    plt.ylabel("predict", family='Times New Roman')
    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')

    plt.text(1, 14, 'R^2=' + str(round(R2, 3)),family='Times New Roman')
    plt.text(1, 15, 'MAE(m)=' + str(round(MAE, 3)),family='Times New Roman')
    plt.text(1, 16, 'MRE(%)=' + str(round(MRE, 3)),family='Times New Roman')
    plt.text(1, 17, 'RMSE(m)=' + str(round(RMSE, 3)) + 'm',family='Times New Roman')


    plt.xlim(0, 18)  # 设置x坐标轴的显示范围
    plt.ylim(0, 18)  # 设置y坐标轴的显示范围
    plt.title(title_name,family = 'Times New Roman')
    plt.colorbar()
    plt.savefig(savename + '.png', bbox_inches='tight', dpi=300)

set_seed()
# input = 'data/machine_dataset.npy'
# data = np.load(input)
# # 获取数组的行数
# num_rows = data.shape[0]

# # 从行索引中随机抽取 10 万个索引
# sample_indices = np.random.choice(num_rows, size=100000, replace=False)

# # 使用抽样的索引从原始数组中获取抽样的行
# data = data[sample_indices, :]
# depth = data[:,2]
# input = data[:,3:7]
# Xtrain,Xtest,Ytrain,Ytest = train_test_split(input,depth,test_size=0.1,random_state=1234)
# # 随机森林
# print('='*10,'RF','='*10)
# model = RandomForestRegressor(n_estimators=1000, max_depth=50, min_samples_split=2,random_state=1234)
# model.fit(Xtrain, Ytrain)
# Yhat = model.predict(Xtrain)
# Y_test = model.predict(Xtest)
# print('训练集R^2: %f' % model.score(Xtrain,Ytrain))
# print('测试集R^2: %f' % model.score(Xtest,Ytest))
# print('RMSE: %f' % mean_squared_error(Ytest,Y_test) ** 0.5)
# print('MAE: %f' % median_absolute_error(Ytest,Y_test))
# print('MAPE: %f' % mean_absolute_percentage_error(Ytest,Y_test))
# scatter_out(Ytest,Y_test,'Random Forest','RF')
# print('='*20)

# # SVR
# print('='*10,'SVR','='*10)
# model =make_pipeline(StandardScaler(), SVR(kernel="rbf",epsilon=0.7))
# model.fit(Xtrain, Ytrain)
# Yhat = model.predict(Xtrain)
# Y_test = model.predict(Xtest)
# print('训练集R^2: %f' % model.score(Xtrain,Ytrain))
# print('测试集R^2: %f' % model.score(Xtest,Ytest))
# print('RMSE: %f' % mean_squared_error(Ytest,Y_test) ** 0.5)
# print('MAE: %f' % median_absolute_error(Ytest,Y_test))
# print('MAPE: %f' % mean_absolute_percentage_error(Ytest,Y_test))
# scatter_out(Ytest,Y_test,'SVR','SVR')
# print('='*20)

# #GBDT
# print('='*10,'GBDT','='*10)
# model = GradientBoostingRegressor(n_estimators=1000,max_depth=50,random_state=1234)
# model.fit(Xtrain, Ytrain)
# Yhat = model.predict(Xtrain)
# Y_test = model.predict(Xtest)
# print('训练集R^2: %f' % model.score(Xtrain,Ytrain))
# print('测试集R^2: %f' % model.score(Xtest,Ytest))
# print('RMSE: %f' % mean_squared_error(Ytest,Y_test) ** 0.5)
# print('MAE: %f' % median_absolute_error(Ytest,Y_test))
# print('MAPE: %f' %mean_absolute_percentage_error(Ytest,Y_test))
# print('='*20)

# #XGB
# print('='*10,'XGBoost','='*10)
# model = XGBRegressor(learning_rate=0.05,eta=0.01,max_depth=50,n_estimators=1000,random_state=1234)
# model.fit(Xtrain, Ytrain)
# Yhat = model.predict(Xtrain)
# Y_test = model.predict(Xtest)
# print('训练集R^2: %f' % model.score(Xtrain,Ytrain))
# print('测试集R^2: %f' % model.score(Xtest,Ytest))
# print('RMSE: %f' % mean_squared_error(Ytest,Y_test) ** 0.5)
# print('MAE: %f' % median_absolute_error(Ytest,Y_test))
# print('MAPE: %f' % mean_absolute_percentage_error(Ytest,Y_test))
# scatter_out(Ytest,Y_test,'XGBoost','XGBoost')
# print('='*20)

input = 'data/batches_16.npy'
input = np.load(input)
# 将第四维的顺序变为 (RGB, depth)
arr = input[:, :, :, [5, 4, 3, 6, 2]]
arr = np.transpose(arr, (0, 3, 1, 2)) #  batch * channels * height * width
# 划分训练集和验证集，其中 test_size=0.1 表示将 10% 的数据分配给验证集
train_data, val_data = train_test_split(arr, test_size=0.1,random_state=1234)
dtype=torch.float32
depth = val_data[:,4]
input = val_data[:,:4]
train_depth = train_data[:,4]
train_input = train_data[:,:4]
train_x = torch.tensor(train_input, dtype=dtype).cuda()
train_y = torch.tensor(train_depth, dtype=dtype).unsqueeze(1) .cuda()

x = torch.tensor(input, dtype=dtype).cuda()
y = torch.tensor(depth, dtype=dtype).unsqueeze(1) .cuda()
# Create model
model = UNET().cuda()

model.eval()
with torch.no_grad():
    state_dict = torch.load('cp/v4-1/950.pth')
    model.load_state_dict(state_dict)

    # 计算训练集R^2
    train_y_hat = model(train_x)
    ss_total = torch.sum((train_y - torch.mean(train_y)) ** 2)
    ss_residual = torch.sum((train_y - train_y_hat) ** 2)
    train_r2 = 1 - (ss_residual / ss_total)   
    print('训练集R^2: %f' % train_r2)

    y_hat = model(x)
    # 计算验证集R^2
    ss_total = torch.sum((y - torch.mean(y)) ** 2)
    ss_residual = torch.sum((y - y_hat) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    print('验证集R^2: %f' % r2)
    # 计算RMSE
    rmse = torch.sqrt(torch.mean((y_hat - y) ** 2))
    # 计算相对差异
    relative_diff = torch.mean(torch.abs(y_hat - y) / y)
    # 计算MAE
    mae = torch.mean(torch.abs(y_hat - y))

    print('RMSE: %f' % rmse)
    print('MAE: %f' % mae)
    print('MRE: %f' % relative_diff)

Ytest = depth.flatten()
Y_test = y_hat.flatten().cpu().numpy()
num_rows = Ytest.shape[0]    
sample_indices = np.random.choice(num_rows, size=10000, replace=False)    
Ytest = Ytest[sample_indices]
Y_test = Y_test[sample_indices]
# scatter_out(Ytest,Y_test,'UNET','UNET')

                    

