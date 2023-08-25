import argparse
import os
import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils
import torchvision.utils as vutils 
from tensorboardX import SummaryWriter
from Unet import UNET
from preprocess.transform import depth_dataset
from loss import ssim, custom_loss_function
from data import getTrainingTestingData
from utils import AverageMeter, colorize, DepthNorm


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def LogProgress(model, writer, test_loader, niter):
    model.eval()
    dtype=torch.cuda.FloatTensor
    sequential = test_loader
    sample_batched = next(iter(sequential))
    image = sample_batched['bands'].cuda().type(dtype)
    depth = sample_batched['depth'].cuda(non_blocking=True).type(dtype).unsqueeze(1)
    output = model(image)
    diff = torch.abs(output-depth).data

    if niter == 0: 
        writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=8, normalize=True), niter)
        writer.add_image('Train.2.Instiu', colorize(vutils.make_grid(depth.data, nrow=8, normalize=False)), niter)
    else:
        writer.add_image('Train.3.Predict', colorize(vutils.make_grid(output.data, nrow=8, normalize=False)), niter)  
        writer.add_image('Train.4.Diff', colorize(vutils.make_grid(diff, nrow=8, normalize=False)), niter)
    del image
    del depth
    del output

def main():
    set_seed(1234)
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=1600, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('--checkpoint', type=str, default='cp',
                     help='In which folder do you want to save the model')
    parser.add_argument('--npy_folder', type=str, default='data/batches_16.npy',
                    help='In which folder do you want to save the model')
    args = parser.parse_args()

    # Create model
    model = UNET().cuda()

    # Create optimizer
    optimizer = torch.optim.Adam( model.parameters(), args.lr )

    # Training parameters
    batch_size = args.bs
    input = args.npy_folder
    prefix = 'UNET_' + str(batch_size)

    # Load data
    dtype=torch.cuda.FloatTensor
    train_loader, val_loader = depth_dataset(input,batch_size)

    # Logging
    writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)

    # Loss
    l1_criterion = nn.L1Loss()

    # Save_folder
    folder_name = os.path.join(args.checkpoint,'v4-1')
    if not os.path.exists(folder_name): os.mkdir(folder_name)

     # Start training...
    for epoch in range(args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)

        # Switch to train mode
        model.train()
        end = time.time()
        
        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['bands'].cuda()).type(dtype)
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True)).type(dtype)

            # depth
            depth_n = depth.unsqueeze(1)
            # Normalize depth
            # depth_n = DepthNorm(depth_n)

            # Predict
            output = model(image.type(dtype))

            # Compute the loss
            l_depth = l1_criterion(output, depth_n)
            l_si = custom_loss_function(output,depth_n)
            loss = (1.0 * l_si) + (0.1 * l_depth)

            # Update step
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))
        
            # Log progress
            niter = epoch*N+i
            if i % 5 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                'ETA {eta}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'
                .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

                # Log to tensorboard
                writer.add_scalar('Train/Loss', losses.val, niter)

            if i % 5 == 0:
                LogProgress(model, writer, val_loader, niter)

        LogProgress(model, writer, val_loader, niter)
        writer.add_scalar('Train/Loss.avg', losses.avg, epoch)

        if epoch % 1 == 0:
            model.eval()
            # Record epoch's intermediate results

            with torch.no_grad():
                for idx, image in enumerate(val_loader):
                    # pdb.set_trace()
                    x = image['bands'].cuda().type(dtype)
                    y = image['depth'].cuda().type(dtype).unsqueeze(1) 
                    y_hat = model(x) 

                    niter_val = epoch*len(val_loader)
                    # 计算loss
                    l_depth = l1_criterion(y_hat, y)
                    l_si = custom_loss_function(y_hat, y)
                    loss = (1.0 * l_si) + (0.1 * l_depth)

                    # 计算RMSE
                    rmse = torch.sqrt(torch.mean((y_hat - y) ** 2))
                    # 计算相对差异
                    relative_diff = torch.mean(torch.abs(y_hat - y) / y)
                    # 计算MAE
                    mae = torch.mean(torch.abs(y_hat - y))
                    
                    # 将结果写入TensorBoard
                    writer.add_scalar('Val/loss', loss.item(), niter_val)  # 将RMSE写入TensorBoard
                    writer.add_scalar('Val/RMSE', rmse.item(), niter_val)  # 将RMSE写入TensorBoard
                    writer.add_scalar('Val/MRE', relative_diff.item(), niter_val)  # 将MRE写入TensorBoard
                    writer.add_scalar('Val/MAE', mae.item(), niter_val)  # 将MAE写入TensorBoard

        if epoch % 50 == 0:
            model_file = os.path.join( folder_name , str(epoch) + ".pth"  )  
            torch.save(model.state_dict(), model_file)


if __name__ == '__main__':
    main()