import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff, PF_loss, cross_entropy_loss
import ipdb
from utils.data_loading import ordinal_encoder


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    bce_score = 0
    pf_score1 = 0
    pf_score2 = 0
    pf_score3 = 0
    pf_score4 = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for idx, batch in enumerate(tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False)):    
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)  # [1, 15, 430, 815] -> [1, 1, 430, 815]

            if net.module.n_classes == 1:
                # ipdb.set_trace()
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                # ipdb.set_trace()
                assert mask_true.min() >= 0 and mask_true.max() < net.module.n_classes, 'True mask indices should be in [0, n_classes)'
                # convert to one-hot format
                # mask_true = F.one_hot(mask_true, net.module.n_classes).permute(0, 3, 1, 2).float()
                # mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.module.n_classes).permute(0, 3, 1, 2).float()
                mask_true_dice = ordinal_encoder(mask_true, net.module.n_classes).permute(1, 0, 2, 3).float()
                mask_pred_dice = ordinal_encoder(mask_pred.argmax(dim=1), net.module.n_classes).permute(1, 0, 2, 3).float()
                    
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred_dice[:, 1:], mask_true_dice[:, 1:], reduce_batch_first=False)

                # bce_loss
                bce_score += cross_entropy_loss(F.one_hot(mask_pred.argmax(dim=1), net.module.n_classes).permute(0, 3, 1, 2).float(), mask_true)

                # compute the PF_loss
                TS, POD, FAR, BIAS = PF_loss(mask_pred.argmax(dim=1), mask_true, net.module.n_classes)
                pf_score1 += TS
                pf_score2 += POD
                pf_score3 += FAR
                pf_score4 += BIAS

    net.train()
    return dice_score / max(num_val_batches, 1), bce_score / max(num_val_batches, 1), pf_score1 / max(num_val_batches, 1), pf_score2 / max(num_val_batches, 1), pf_score3 / max(num_val_batches, 1), pf_score4 / max(num_val_batches, 1)
