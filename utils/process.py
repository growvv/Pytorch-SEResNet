import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# 读取所有的类别
def read_category(base_path, group):
    path = os.path.join(base_path, group)

    # x: categoryname_group_id.png
    categories = [x.split("_")[0] for x in os.listdir(path)]
    categories = list(set(categories))
    categories.sort()
    cat_to_id = dict(zip(categories, range(len(categories))))

    nums = [x.split(".")[0].split("_")[-1] for x in os.listdir(path)]
    num = len(set(nums))

    return categories, cat_to_id, num

def test_read_category():
    base_path = '../data/radar/'
    group = '100001'

    categories, cat_to_id, nums = read_category(base_path, group)

    print(categories)
    print(cat_to_id)
    print(nums)


# 将不同类别的图片在同一个维度堆叠拼接
def stack_img(base_path, group):
    # return: [N, C, H, W]

    categories, cat_to_id, num = read_category(base_path, group)

    imgs = []
    for id in tqdm(range(1, num+1)):
        img = []
        for cat in categories:
            if cat == 'ob':
                continue
            path = os.path.join(base_path, group, cat + "_" + group + "_" + str(id) + ".png")
            
            img.append(cv2.imread(path, cv2.IMREAD_UNCHANGED))

        # 把 "ob" 放到最后一个通道
        path = os.path.join(base_path, group, "ob" + "_" + group + "_" + str(id) + ".png")
        img.append(cv2.imread(path, cv2.IMREAD_UNCHANGED))

        # x = np.stack(img, axis=0)
        # print(x.shape, x.min(), x.max())

        # len* (h, w) -> (h, w, len)
        imgs.append(np.stack(img, axis=0))

    return np.stack(imgs, axis=0)

def test_stack_img():
    base_path = '../data/radar/'
    group = '100001'

    img = stack_img(base_path, group)

    print(img.shape)


# 堆叠多个group的图片
def stack_group(base_path, groups: list):
    # return: [N, C, H, W]

    imgs = []
    for group in groups:
        imgs.append(stack_img(base_path, group))

    return np.concatenate(imgs, axis=0)


def test_stack_group():
    base_path = '../data/radar/'
    groups = ['100001', '100002', '100003']

    img = stack_group(base_path, groups)

    print(img.shape)

# 将图片按group和编号分开保存成.npy文件
def save_img(base_path, groups: list):
    for group in groups:
        img = stack_img(base_path, group)
        np.save(os.path.join(base_path, group + ".npy"), img)

if __name__ == '__main__':
    # test_read_category()
    test_stack_img()
    # test_stack_group()