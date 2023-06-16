import numpy as np
from tqdm import tqdm
# plt
import matplotlib.pyplot as plt

def process(ob):
    # 把ob中的异常值255转换为0
    ob[ob == 255] = 0

    # [0,250] 恢复到原来的范围 [0, 20]     
    ob = ob / 250 * 20

    # 划分成4个等级[0,0.1], [0.1, 3], [3, 10], [10, 20]

    # print(ob)

    ob[ob < 0.1] = 0
    ob[(ob >= 0.1) & (ob < 3)] = 1
    ob[(ob >= 3) & (ob < 10)] = 2
    ob[(ob >= 10)] = 3 

    return ob


def process_radar1(ob):
    # 把ob中的异常值255转换为0
    ob[ob == 255] = 0

    # [0,250] 恢复到原来的范围 [0, 20]     
    ob = ob / 250 * 20

    # 划分成4个等级[0,0.1], [0.1, 3], [3, 10], [10, 20]

    # print(ob)

    a = np.array([0, 0.1, 3, 10, 20])
    a = R_Z_ratio1(a)
    
    ob[ob < a[1]] = 0
    ob[(ob >= a[1]) & (ob < a[2])] = 1
    ob[(ob >= a[2]) & (ob < a[3])] = 2
    ob[(ob >= a[3])] = 3 

    return ob

def process_radar2(ob):
    # 把ob中的异常值255转换为0
    ob[ob == 255] = 0

    # [0,250] 恢复到原来的范围 [0, 20]     
    ob = ob / 250 * 20

    # 划分成4个等级[0,0.1], [0.1, 3], [3, 10], [10, 20]

    # print(ob)

    a = np.array([0, 0.1, 3, 10, 20])
    a = R_Z_ratio2(a)
    
    ob[ob < a[1]] = 0
    ob[(ob >= a[1]) & (ob < a[2])] = 1
    ob[(ob >= a[2]) & (ob < a[3])] = 2
    ob[(ob >= a[3])] = 3 

    return ob

# Z = 3000R^{1.4}
def Z_R_ratio1(radar_img):
    return 3000 * np.power(radar_img, 1.4)

# Z = 200R^1.6
def Z_R_ratio2(radar_img):
    return 200 * np.power(radar_img, 1.6)

def R_Z_ratio1(radar_img):
    return np.power(radar_img / 3000, 1 / 1.4)

def R_Z_ratio2(radar_img):
    return np.power(radar_img / 200, 1 / 1.6)



def cal():
    path = "/home/lfr/mntc/Pytorch-UNet/res/metrice_load414_100_2/checkpoint/npy/"
    for i in tqdm(range(10)):
        true_masks = np.load(f'{path}{i}_true.npy')[0]
        # copy_masks = np.load(f'{path}{i}_copy.npy')[0]
        masks_pred = np.load(f'{path}{i}_pred.npy')[0]
        radar_img = np.load(f'{path}{i}_image.npy')[0][13]
        zr_pred1 = R_Z_ratio1(radar_img)
        zr_pred2 = R_Z_ratio2(radar_img)

        print(zr_pred1.min(), zr_pred1.max())
        print(zr_pred2.min(), zr_pred2.max())

        zr_pred1 = process_radar1(zr_pred1)
        zr_pred1 = process_radar2(zr_pred2)

        # 拼图
        imgs = np.concatenate([zr_pred1, zr_pred1], axis=1)

        # 保存成图片
        plt.imsave(f'zr/{i}.png', imgs)
        
        print(true_masks.shape, masks_pred.shape)

if __name__ == '__main__':
    cal()