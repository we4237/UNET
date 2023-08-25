import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from osgeo import osr, gdal, gdalconst
from torch.utils.data import Dataset, DataLoader

# 创建自定义的数据集类
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sample = self.data[index]

        # 构建字典，表示样本数据
        sample_dict = {
            'bands': sample['bands'],  # 图像数据
            'depth': sample['depth']   # 深度数据
        }
        return sample_dict

    def __len__(self):
        return len(self.data)

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def tranform(dataset,points):
    trans = dataset.GetGeoTransform()
    x = points[:, 0]
    y = points[:, 1]
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b_list = np.array([x - trans[0], y - trans[3]])
    coords = np.empty((len(x),2))
    for i,b in enumerate(b_list.T):
        res = np.linalg.solve(a, b)
        coords[i] = res

    cols = np.array(coords[:,0].astype(np.int32))
    rows = np.array(coords[:,1].astype(np.int32))

    return cols,rows

def depth_dataset(input:str = '', batch_size:int = 32):

    input = np.load(input)
    # 将第五维的顺序变为 (RGB,NIR, depth)
    arr = input[:, :, :, [5, 4, 3, 6, 2]]
    arr = np.transpose(arr, (0, 3, 1, 2)) # 3288 batch * channels * height * width
    # 划分训练集和验证集，其中 test_size=0.1 表示将 10% 的数据分配给验证集
    train_data, val_data = train_test_split(arr, test_size=0.1,random_state=1234)
    
    train_list = []
    for i in range(train_data.shape[0]):
        sample_dict = {
            'bands': train_data[i, :4, :, :],
            'depth': train_data[i, 4, :, :]    
        }
        train_list.append(sample_dict)

    val_list = []
    for i in range(val_data.shape[0]):
        sample_dict = {
            'bands': val_data[i, :4, :, :],  
            'depth': val_data[i, 4, :, :]    
        }
        val_list.append(sample_dict)

    # 创建数据集对象
    train_dataset = CustomDataset(train_list)
    val_dataset = CustomDataset(val_list)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,num_workers=4)

    return train_dataloader,val_dataloader

def save(input:str = ''):
    input_txt = np.loadtxt('/mnt/user/chenmuyin/depth/UNET_cmy/data/downsampled_data.txt')

    # 读取遥感数据
    ds_sample = gdal.Open(input)  # type: gdal.Datasetr
    sample_geotrans = ds_sample.GetGeoTransform()
    sample_proj = ds_sample.GetProjection()
    sample_width = ds_sample.RasterXSize
    sample_height = ds_sample.RasterYSize
    sample_bands = ds_sample.RasterCount
    sample_data = ds_sample.ReadAsArray(0, 0, sample_width, sample_height)
    im_data = np.nan_to_num(sample_data)

    gdal.AllRegister()
    labels = abs(input_txt[:,2])
    Points = (np.array([input_txt[:,0], input_txt[:,1]])).T
    cols,rows = tranform(ds_sample,Points)
    new_arr = np.column_stack((cols,rows,labels))
    input = [im_data[:4, row, col] for row,col in zip(rows,cols)]

    data = np.concatenate((new_arr, input), axis=1)


    # # 将数据按行号升序、列号升序排序
    # data = data[np.lexsort((data[:, 1], data[:, 0]))]    
    # 切片成 16*16 的 batch，并保存在一个列表中
    block_size = 16
    num_batches = len(data) // block_size**2
    batches = []
    for i in range(num_batches):
        start = i * block_size**2
        end = start + block_size**2
        batch = data[start:end].reshape(block_size, block_size, -1)
        batches.append(batch)

    # 将 batch 列表保存为 numpy 数组
    batches = np.array(batches)
    
    # 保存数组为 npy 文件
    np.save('data/batches_removed_16.npy', batches)

def crop(source_file = '/mnt/user/chenmuyin/depth/UNET_cmy/data/resampled01/ganquan', 
               target_file = 'data/cropped/cropped'):
    dataset = gdal.Open(source_file, gdalconst.GA_ReadOnly)
    sample_width = dataset.RasterXSize
    sample_height = dataset.RasterYSize
    sample_data = dataset.ReadAsArray(0, 0, sample_width, sample_height)
    im_data = np.nan_to_num(sample_data)
    band_count = dataset.RasterCount  # 波段数
    geotrans = list(dataset.GetGeoTransform())

    if os.path.exists(target_file) and os.path.isfile(target_file):  # 如果已存在同名影像
        os.remove(target_file)  # 则删除之

    input_txt = np.loadtxt('/mnt/user/chenmuyin/depth/UNET_cmy/data/downsampled_data.txt')
    Points = (np.array([input_txt[:,0], input_txt[:,1]])).T
    cols,rows = tranform(dataset,Points)

    row_max,row_min = rows.max(),rows.min()
    col_max,col_min = cols.max(),cols.min()

    cropped_image = im_data[:,row_min:row_max, col_min:col_max]

    band1 = dataset.GetRasterBand(1)
    data_type = band1.DataType
    cols = cropped_image.shape[2]
    rows = cropped_image.shape[1]
    target = dataset.GetDriver().Create(target_file, xsize=cols, ysize=rows, bands=band_count,
                                        eType=data_type)
    sample_proj = dataset.GetProjection()
    target.SetProjection(sample_proj)  # 设置投影坐标
    target.SetGeoTransform(geotrans)  # 设置地理变换参数
    for i in range(band_count):

        out_band  = target.GetRasterBand(i+1)
        out_band.WriteArray(cropped_image[i, :, :])
        out_band.FlushCache()
        out_band.ComputeBandStats(False)  # 计算统计信息
    # print("正在写入完成")
    del dataset
    del target

def slice(input,block_size=16):

    # 读取遥感图像
    ds_sample = gdal.Open(input)
    sample_data = ds_sample.ReadAsArray()

    # 提取蓝绿红近四个波段
    blue_band = sample_data[0]  # 索引为1的波段为蓝波段
    green_band = sample_data[1]  # 索引为0的波段为绿波段
    red_band = sample_data[2]  # 索引为2的波段为红波段
    NIR_band = sample_data[3]

    image = np.stack((red_band, green_band, blue_band,NIR_band), axis=2)
    height, width, bands = image.shape
    image = image[:height - height % block_size, :width - width % block_size, :]

    image = image.transpose(2,0,1)


    # 切分为多个小块
    num_blocks_h = image.shape[1] // block_size
    num_blocks_w = image.shape[2] // block_size
    blocks = []
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block = image[:, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            blocks.append(block)

    # 保存为numpy数组
    blocks_array = np.array(blocks)

    # 保存到npy文件
    np.save('data/blocks_removed.npy', blocks_array)


if __name__ == '__main__':
    save()
    # crop()
    slice()