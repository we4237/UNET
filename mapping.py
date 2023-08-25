import os
import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from osgeo import osr, gdal, gdalconst
from torch.utils.data import Dataset, DataLoader
from Unet import UNET

class MapDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sample = self.data[index]
        return sample

    def __len__(self):
        return len(self.data)



if __name__ == '__main__':

    batch_size = 128
    input = 'data/blocks.npy'   
    input = np.load(input)

    # 创建数据集对象
    map_dataset = MapDataset(input)

    # 创建数据加载器
    map_dataloader = DataLoader(map_dataset, batch_size=batch_size, shuffle=False,num_workers=8)

    dtype=torch.cuda.FloatTensor

    # Create model
    model = UNET().cuda()

    model.eval()
    state_dict = torch.load('cp/v4-1/1000.pth')
    model.load_state_dict(state_dict)

    predictions = []
    for i,batch in enumerate(map_dataloader):
        with torch.no_grad():
            x = torch.autograd.Variable(batch.cuda()).type(dtype)
            y_pred = model(x)

            # 将当前批次的预测结果添加到列表中
            predictions.extend(y_pred)

    # 将预测结果转换为一个总的列表，长度为 len(batch) * batch_size
    predictions = torch.cat(predictions, dim=0)
    # 保存 Tensor 到文件
    # torch.save(predictions, 'data/predictions_.pt')

    # 加载保存的 Tensor
    # predictions = torch.load('data/predictions_.pt')
    num_blocks_h = 1282
    num_blocks_w = 679
    block_size = 16
    depth_map = np.zeros((num_blocks_h*block_size, num_blocks_w*block_size))
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            block_depths = predictions[i * num_blocks_w + j].cpu().numpy()
            depth_map[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = block_depths

    np.save('output/depth_1000.npy',depth_map)

