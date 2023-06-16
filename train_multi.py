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


def train_model(
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
    print(f'len(dataset) = {len(dataset)}')
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)  # suffle=Flase时才能使用
    train_loader = DataLoader(train_set, shuffle=False, sampler=train_sampler,  **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(
    #     dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #          val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    # )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
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

    writer = SummaryWriter(log_dir=args.log_dir, comment='train')

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_wbce_loss = 0
        epoch_wdice_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for idx, batch in enumerate(train_loader):
                # if idx > 2:
                #     break
                images, true_masks = batch['image'], batch['mask']  # [1, 15, 430, 815] 和 [1, 430, 815]

                assert images.shape[1] == model.module.n_channels, \
                    f'Network has been defined with {model.module.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                # 四个等级，weight=1，4，8，32
                weights = torch.tensor([1, 4, 8, 32]).to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images) # [1, 15, 430, 815] -> [1, 1, 430, 815]
                    # ipdb.set_trace()
                    if model.module.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float()) # [1, 430, 815]
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        # loss = criterion(weights, masks_pred, true_masks)
                        # # loss += dice_loss(
                        # #     F.softmax(masks_pred, dim=1).float(),
                        # #     F.one_hot(true_masks, model.module.n_classes).permute(0, 3, 1, 2).float(),
                        # #     multiclass=True
                        # # )
                        # loss += dice_loss(
                        #     F.softmax(masks_pred, dim=1).float(),  # [1, 4, 430, 815]
                        #     ordinal_encoder(true_masks, model.module.n_classes).permute(1, 0, 2, 3).float(),
                        #     multiclass=True
                        # )
                        # masks_pred: [B, 4, 430, 815], true_masks: [B, 430, 815], weights: [4]
                        loss = weight_cross_entropy_loss(masks_pred, true_masks, weights)
                        wbce_loss = loss.item()
                        loss += weight_dice_loss(
                            F.softmax(masks_pred, dim=1).float(),  # [1, 4, 430, 815]
                            ordinal_encoder(true_masks, model.module.n_classes).permute(1, 0, 2, 3).float(),
                            weights,
                        )
                        wdice_loss = loss.item() - wbce_loss

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                epoch_wbce_loss += wbce_loss
                epoch_wdice_loss += wdice_loss
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                writer.add_scalar('train batch loss', loss.item(), global_step)
                writer.add_scalar('train batch wbce loss', wbce_loss, global_step)
                writer.add_scalar('train batch wdice loss', wdice_loss, global_step)
                
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Evaluation round
        # val_score, bce_loss, TS_loss, Pod_loss, FAR_loss, BAIS_loss = evaluate(model, val_loader, device, amp)
        # mock evaluate
        val_score, bce_loss, TS_loss, Pod_loss, FAR_loss, BAIS_loss = 0, 0, 0, 0, 0, 0
        scheduler.step(val_score)

        # ipdb.set_trace()

        # logging.info('Validation Dice score: {}'.format(val_score))
        # logging.info('Validation BCE loss: {}'.format(bce_loss))
        # logging.info('Validation TS loss: {}'.format(TS_loss))
        # logging.info('Validation Pod loss: {}'.format(Pod_loss))
        # logging.info('Validation FAR loss: {}'.format(FAR_loss))
        # logging.info('Validation BAIS loss: {}'.format(BAIS_loss))
        # cv2.imwrite('res/images/{}_true.png'.format(global_step), true_masks[0].float().cpu().numpy())
        # cv2.imwrite('res/images/{}_pred.png'.format(global_step), masks_pred.argmax(dim=1)[0].float().cpu().numpy())
        np.save(args.dir_checkpoint + 'images/{}_true.npy'.format(epoch), true_masks[0].float().cpu().numpy())
        np.save(args.dir_checkpoint + 'images/{}_pred.npy'.format(epoch), masks_pred.argmax(dim=1)[0].float().cpu().numpy())
        # images
        np.save(args.dir_checkpoint + 'images/{}_image.npy'.format(epoch), images[0].float().cpu().numpy())

        # use tensorboard to visualize the results
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('validation Dice', val_score, epoch)
        writer.add_scalar('validation BCE loss', bce_loss, epoch)
        writer.add_scalar('validation TS loss', TS_loss, epoch)
        writer.add_scalar('validation Pod loss', Pod_loss, epoch)
        writer.add_scalar('validation FAR loss', FAR_loss, epoch)
        writer.add_scalar('validation BAIS loss', BAIS_loss, epoch)
        writer.add_scalar('train epoch loss', epoch_loss / len(train_loader), epoch)
        writer.add_scalar('train epoch wbce loss', epoch_wbce_loss / len(train_loader), epoch)
        writer.add_scalar('train epoch wdice loss', epoch_wdice_loss / len(train_loader), epoch)

        logging.info('Epoch {} finished ! Loss: {}'.format(epoch, epoch_loss / len(train_loader)))

        if save_checkpoint and epoch % args.save_interval == 0:
            Path(args.dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, args.dir_checkpoint + 'checkpoint_epoch{}.pth'.format(epoch))
            logging.info(f'Checkpoint {epoch} saved!')


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


    train_model(
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
