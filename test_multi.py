import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import ipdb
import numpy as np

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset, RadarDataset, ordinal_encoder
from utils.dice_score import dice_loss, weight_dice_loss, weight_cross_entropy_loss
from utils.utils import checkdir, plot_img_and_mask
from utils.metrices import precision, recall, ACC, FSC, HSS, BIAS
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random
from torch.utils.tensorboard import SummaryWriter

# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
# dir_checkpoint = Path('./checkpoints/')


def test_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        **kwargs
):
    # 1. Create dataset
    try:
        # dataset = CarvanaDataset(args.dir_img, args.dir_mask, img_scale)
        dataset = RadarDataset(args.dir_img, args.dir_mask)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(args.dir_img, args.dir_mask, img_scale)

    # 2. Split into train / validation partitions

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)  # suffle=Flase时才能使用
    test_loader = DataLoader(dataset, shuffle=False, **loader_args)


    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Testing size:    {len(dataset)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.module.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    precision_a = 0
    recall_a = 0
    ACC_a = 0
    FSC_a = 0
    HSS_a = 0
    BIAS_a = 0


    # Test model
    model.eval()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing'):
            images, true_masks, copy_masks = batch['image'], batch['mask'], batch['mask_c']  # [1, 15, 430, 815] 和 [1, 430, 815]                
            assert images.shape[1] == model.module.n_channels, \
                f'Network has been defined with {model.module.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'                
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            # 四个等级，weight=1，4，8，32
            weights = torch.tensor([1, 4, 8, 32]).to(device=device, dtype=torch.float32)                
            masks_pred = model(images) # [1, 15, 430, 815] -> [1, 1, 430, 815]
            
            # ipdb.set_trace()
            true_masks = true_masks.float().cpu().numpy()  # (B, 430, 815)
            masks_pred = masks_pred.argmax(dim=1).float().cpu().numpy()  # (B, 430, 815)
            images = images.float().cpu().numpy()  # (B, 15, 430, 815)

            # metrics
            # ipdb.set_trace()
            precision_v = precision(true_masks, masks_pred)
            recall_v = recall(true_masks, masks_pred)
            ACC_v = ACC(true_masks, masks_pred)
            FSC_v = FSC(true_masks, masks_pred)
            HSS_v = HSS(true_masks, masks_pred)
            BIAS_v = BIAS(true_masks, masks_pred)

            precision_a += precision_v
            recall_a += recall_v
            ACC_a += ACC_v
            FSC_a += FSC_v
            HSS_a += HSS_v
            BIAS_a += BIAS_v

            # save
            np.save(args.dir_checkpoint + 'npy/{}_true.npy'.format(idx), true_masks)
            np.save(args.dir_checkpoint + 'npy/{}_pred.npy'.format(idx), masks_pred)
            np.save(args.dir_checkpoint + 'npy/{}_copy.npy'.format(idx), copy_masks)
            np.save(args.dir_checkpoint + 'npy/{}_image.npy'.format(idx), images)
            
            # save as image
            save_images(masks_pred, true_masks, copy_masks, images, idx)



    model.train()

    precision_a /= len(test_loader)
    recall_a /= len(test_loader)
    ACC_a /= len(test_loader)
    FSC_a /= len(test_loader)
    HSS_a /= len(test_loader)
    BIAS_a /= len(test_loader)

    logging.info('Tested: precision: {:.4f}, recall: {:.4f}, ACC: {:.4f}, FSC: {:.4f}, HSS: {:.4f}, BIAS: {:.4f}'.format(
        precision_a, recall_a, ACC_a, FSC_a, HSS_a, BIAS_a))

    logging.info('Model tested and saved to {}'.format(args.dir_checkpoint))

def save_images(masks_pred, true_masks, copy_masks, images, idx):
    # masks_pred: [B, 430, 815], true_masks: [B, 430, 815], images: [B, 15, 430, 815]
    # save as image
    for i in range(masks_pred.shape[0]):
        mask_pred = masks_pred[i]  # (430, 815)
        true_mask = true_masks[i] # (430, 815)
        copy_mask = copy_masks[i] # (430, 815)
        image = images[i]  # (15, 430, 815)
        # save as image
        import matplotlib.pyplot as plt
        plt.imsave(args.dir_checkpoint + 'images/{}_true.png'.format(idx), true_mask, cmap='gray')
        plt.imsave(args.dir_checkpoint + 'images/{}_pred.png'.format(idx), mask_pred, cmap='gray')
        plt.imsave(args.dir_checkpoint + 'images/{}_copy.png'.format(idx), copy_mask, cmap='gray')
        for j in range(image.shape[0]):
            plt.imsave(args.dir_checkpoint + 'images/{}_image_{}.png'.format(idx, j), image[j], cmap='gray')

        # cat mask_pred, true_mask, image[13], 按照宽拼接
        thr_img = np.concatenate((mask_pred, true_mask), axis=1)
        plt.imsave(args.dir_checkpoint + 'cat/{}_cat.png'.format(idx), thr_img, cmap='gray')
        # 保存时 不改变值的范围

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--in_classes', '-inc', type=int, default=15, help='Number of in classes')
    parser.add_argument('--out_classes', '-outc', type=int, default=1, help='Number of out classes')

    parser.add_argument('--dir_img', type=str, default='data/radar_npy/factors/', help='Directory of the images')
    parser.add_argument('--dir_mask', type=str, default='data/radar_npy/ob/', help='Directory of the masks')
    # save_checkpoint
    # save_interval
    parser.add_argument('--save_interval', type=int, default=10, help='save checkpoint interval (default: 10)')
    parser.add_argument('--save_checkpoint', type=bool, default=False, help='Directory of the masks')
    parser.add_argument('--dir_checkpoint', type=str, default='checkpoints/', help='Directory of the checkpoints')
    # log_dir
    parser.add_argument('--log_dir', type=str, default='runs/', help='Directory of the logs')

    # ddp
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--master_addr", default="127.0.0.1", type=str)
    parser.add_argument("--master_port", default="12355", type=str)

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    Path(args.dir_checkpoint).mkdir(parents=True, exist_ok=True)
    Path(args.dir_checkpoint + 'images/').mkdir(parents=True, exist_ok=True)
    Path(args.dir_checkpoint + 'npy/').mkdir(parents=True, exist_ok=True)   
    Path(args.dir_checkpoint + 'cat/').mkdir(parents=True, exist_ok=True)    
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # 设置随机数种子，以确保每次运行的结果都相同
    torch.manual_seed(0)

    # 定义多 GPU 模型并设置
    torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    print("lock rank: ", local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda:{}".format(local_rank))

    model = UNet(n_channels=args.in_classes, n_classes=args.out_classes, bilinear=args.bilinear).to(device)
    model = model.to(memory_format=torch.channels_last)
    model = model.to(device=device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    
    logging.info(f'Network:\n'
                 f'\t{args.in_classes} input channels\n'
                 f'\t{args.out_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if args.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        # del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')


    test_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        val_percent=args.val / 100,
        save_checkpoint=args.save_checkpoint,
        amp=args.amp,
        args=args,
    )
