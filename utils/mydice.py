import torch
import torch.nn as nn

def weight_dice_coeff(input, target, weight):
    
    eps = 1e-6   # 避免分母为 0     
    intersections = torch.sum(input * target * weight.view(1, -1, 1, 1), dim=(1,2,3))     
    unions = torch.sum(input * weight.view(1, -1, 1, 1) + target * weight.view(1, -1, 1, 1), dim=(1,2,3))     
    dice = (2 * intersections + eps) / (unions + eps)   
    print(dice.shape) 
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

if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    input = torch.rand(1, 4, 430, 815)
    target = torch.randint(0, 4, (1, 1, 430, 815))
    weight = torch.rand(4)
    print(input.shape, target.shape, weight.shape)
    print(weight_cross_entropy_loss(input, target, weight))


