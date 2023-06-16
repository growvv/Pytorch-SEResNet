import torch
from torch import Tensor
import torch.nn.functional as F
import ipdb 


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def weight_dice_coeff(input, target, weight):
    
    eps = 1e-6   # 避免分母为 0     
    intersections = torch.sum(input * target * weight.view(1, -1, 1, 1), dim=(1,2,3))     
    unions = torch.sum(input * weight.view(1, -1, 1, 1) + target * weight.view(1, -1, 1, 1), dim=(1,2,3))     
    dice = (2 * intersections + eps) / (unions + eps)   
    # print(dice.shape) 
    return dice.mean()

def weight_dice_loss(input, target, weight):
    return 1 - weight_dice_coeff(input, target, weight)

def weight_cross_entropy_loss(input, target, weight):
    
    N, C, H, W = input.shape                      # 获取输入大小

    input = input.view(N, C, -1)                  # 将输入大小转换成[N, C, H*W]
    target = target.view(N, -1)                   # 将目标大小转换成[N, H*W]

    # 使用交叉熵损失函数计算损失
    loss = torch.nn.functional.cross_entropy(input, target, weight=weight, reduction='mean')
    
    return loss

# 分类的损失函数
def cross_entropy_loss(input: Tensor, target: Tensor, weight: Tensor = None, ignore_index: int = -100):
    # Cross entropy loss (objective to minimize) between 0 and 1

    if input.size(1) == 1:
        return F.binary_cross_entropy_with_logits(input.squeeze(1), target.float(), weight, reduction='mean', pos_weight=weight)
    else:
        return F.cross_entropy(input, target, weight, ignore_index=ignore_index, reduction='mean')
    

def test_cross_entropy_loss():
    input = torch.rand(1, 4, 430, 815)
    input = F.one_hot(input.argmax(dim=1), 4).permute(0, 3, 1, 2).float()
    print("input: ", input.shape, input.min(), input.max())
    target = torch.randint(0, 4, (1, 430, 815))

    weight = torch.rand(1, 4)
    print(cross_entropy_loss(input, target, weight))

# TP (True positive) FN (False negative) FP (False positive) TN (True negative)
# input: Forecast, target: truth
def confusion_matrix(input: Tensor, target: Tensor, n_classes: int, ignore_index: int = -100):
    # input: [B, H, W], target: [B, H, W]

    assert input.size() == target.size()

    input[input<n_classes//2] = 0
    input[input>=n_classes//2] = 1

    target[target<n_classes//2] = 0
    target[target>=n_classes//2] = 1
    
    TP = sum(((input == 1) & (target == 1)).reshape(-1))
    FP = sum(((input == 1) & (target == 0)).reshape(-1))
    FN = sum(((input == 0) & (target == 1)).reshape(-1))
    TN = sum(((input == 0) & (target == 0)).reshape(-1))

    assert TP + FP + FN + TN == input.numel()

    return TP, FP, FN, TN


def test_confusion_matrix():
    input = torch.rand(1, 4, 3, 3)
    input = input.argmax(dim=1)
    target = torch.randint(0, 4, (1, 3, 3))

    confusion_matrix(input, target, 4)

# TS, POD 越大越好, FAR, BIAS 越小越好
def PF_loss(input: Tensor, target: Tensor, n_classes: int, ignore_index: int = -100):
    # input: [B, H, W], target: [B, H, W]
    assert input.size() == target.size()

    TP, FP, FN, TN = confusion_matrix(input, target, n_classes, ignore_index)
    TS = TP / (TP + FP + FN + 1e-6)
    POD = TP / (TP + FN + 1e-6)
    FAR = FP / (FP + TP + 1e-6)
    BIAS = (TP + FP) / (TP + FN + 1e-6)

    return TS, POD, FAR, BIAS

def test_PF_loss():
    input = torch.rand(1, 4, 430, 815)
    input = input.argmax(dim=1)
    target = torch.randint(0, 4, (1, 430, 815))

    TS, POD, FAR, BIAS = PF_loss(input, target, 4)
    print(TS, POD, FAR, BIAS)


if __name__ == "__main__":
    # test_cross_entropy_loss()
    # test_confusion_matrix()
    # test_PF_loss()
    # test_weight_dice_loss()
    pass
