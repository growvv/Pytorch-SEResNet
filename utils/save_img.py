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


# 将不同类别的图片在同一个维度堆叠拼接, 并保存成.npy文件
def save_img(base_path, save_path, group):
    # return: N*[C, H, W]

    categories, cat_to_id, num = read_category(base_path, group)

    for id in tqdm(range(1, num+1)):
        img = []
        for cat in categories:
            if cat == 'ob':
                continue
            path = os.path.join(base_path, group, cat + "_" + group + "_" + str(id) + ".png")
            
            img.append(cv2.imread(path, cv2.IMREAD_UNCHANGED))

        # 分别保存因子和降水
        factors = np.stack(img, axis=0)
        np.save(os.path.join(os.path.join(save_path, "factors"), group + "_" + str(id) + ".npy"), factors)

        path = os.path.join(base_path, group, "ob" + "_" + group + "_" + str(id) + ".png")
        ob_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        np.save(os.path.join(os.path.join(save_path, "ob"), group + "_" + str(id) + ".npy"), ob_img)
    

def test_save_img():
    base_path = '../data/radar/'
    group = '100001'
    save_path = '../data/radar_npy/'

    save_img(base_path, save_path, group)


def test_save_img2():
    base_path = '../data/radar/'
    save_path = '../data/radar_npy/'

    groups = sorted(os.listdir(base_path), key=lambda x: int(x))
    for group in groups:
        # print(group)
        save_img(base_path, save_path, group)

def test_read_npy():
    base_path = '../data/radar_npy/ob/'
    group = '100001'
    id = 1

    path = os.path.join(base_path, group + "_" + str(id) + ".npy")

    img = np.load(path)
    print(img.shape)


if __name__ == '__main__':
    # test_save_img()
    # test_save_img2()

    # base_path = '../data/radar/'
    # save_path = '../data/radar_npy/'
    # group = '100005'
    # save_img(base_path, save_path, group)
    test_read_category()

    # test_read_npy()
