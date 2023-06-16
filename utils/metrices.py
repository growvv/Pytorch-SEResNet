# import torch
# from torch import Tensor
# import ipdb 

import numpy as np
# import scipy
# from keras.applications.inception_v3 import InceptionV3


def prep_clf(obs, pre, threshold=1):
    """
    func: 计算二分类结果-混淆矩阵的四个元素
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        hits, misses, falsealarms, correctnegatives
        #aliases: TP, FN, FP, TN
    """
    # 根据阈值分类为 0, 1
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))

    return hits, misses, falsealarms, correctnegatives

# FAR
def precision(obs, pre, threshold=0.1, eps=1e-6):
    """
    func: 计算精确度precision: TP / (TP + FP)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        dtype: float
    """

    TP, FN, FP, TN = prep_clf(obs=obs, pre=pre, threshold=threshold)

    return TP / (TP + FP + eps)


# POD
def recall(obs, pre, threshold=0.1, eps=1e-6):
    """
    func: 计算召回率recall: TP / (TP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        dtype: float
    """

    TP, FN, FP, TN = prep_clf(obs=obs, pre=pre, threshold=threshold)

    return TP / (TP + FN + eps)


def ACC(obs, pre, threshold=0.1, eps=1e-6):
    """
    func: 计算准确度Accuracy: (TP + TN) / (TP + TN + FP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        dtype: float
    """

    TP, FN, FP, TN = prep_clf(obs=obs, pre=pre, threshold=threshold)

    return (TP + TN) / (TP + TN + FP + FN + eps)


def FSC(obs, pre, threshold=0.1, eps=1e-6):
    """
    func:计算f1 score = 2 * ((precision * recall) / (precision + recall))
    """
    precision_socre = precision(obs, pre, threshold=threshold)
    recall_score = recall(obs, pre, threshold=threshold)

    return 2 * ((precision_socre * recall_score) / (precision_socre + recall_score + eps))


def HSS(obs, pre, threshold=0.1, eps=1e-6):
    """
    HSS - Heidke skill score
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): pre
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: HSS value
    """
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           threshold=threshold)

    HSS_num = 2 * (hits * correctnegatives - misses * falsealarms)
    HSS_den = (misses ** 2 + falsealarms ** 2 + 2 * hits * correctnegatives +
               (misses + falsealarms) * (hits + correctnegatives))

    return HSS_num / (HSS_den + eps)


def BIAS(obs, pre, threshold=0.1):
    '''
    func: 计算Bias评分: Bias =  (hits + falsealarms)/(hits + misses)
    	  alias: (TP + FP)/(TP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           threshold=threshold)

    return (hits + falsealarms) / (hits + misses + 1e-6)


def TS(obs, pre, threshold=0.1):
    '''
    func: 计算Ts评分: Ts =  hits/(hits + misses + falsealarms)
    	  alias: TP/(TP + FN + FP)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           threshold=threshold)

    return hits / (hits + misses + falsealarms + 1e-6)


# # load pretrained model and calculate FID from 2 images
# def calculate_fid(real_images, fake_images):
#     """
#     The FID metric calculates the distance between two distributions of images.
#     :param real_images: real images from dataset, numpy array
#     :param fake_images: generated images from G, numpy array
#     :return: the FID score
#     """
#     # import pretrained model for calculate FID, input shape is 64*64*1
#     model = InceptionV3(include_top=False, pooling='avg', input_shape=(64, 64, 3))

#     act1 = model.predict(real_images)
#     act2 = model.predict(fake_images)
#     mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
#     mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
#     ssdiff = np.sum((mu1 - mu2) ** 2.0)
#     covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
#     fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
#     return fid

# TP (True positive) FN (False negative) FP (False positive) TN (True negative)
# input: Forecast, target: truth
def confusion_matrix(input: np.array, target: np.array, threshold: int = 1):
    # input: [B, H, W], target: [B, H, W], range: [0, 3]
    print(input.shape, target.shape)

    assert input.shape == target.shape

    # ipdb.set_trace()

    B, H, W = input.shape
    # input = input.reshape(-1)
    # target = target.reshape(-1)

    input = np.where(input >= threshold, 1, 0)
    target = np.where(target >= threshold, 1, 0)

    TP = np.sum((input == 1) & (target == 1))
    FP = np.sum((input == 1) & (target == 0))
    FN = np.sum((input == 0) & (target == 1))
    TN = np.sum((input == 0) & (target == 0))

    # assert torch.sum(TP + FP + FN + TN) == input.numel()
    

    return TP, FP, FN, TN

    

if __name__ == "__main__":
    # [B, 2, 3] np.array, range: [0, 3]
    input = np.array([[[0, 1, 2], [0, 3, 2]], [[0, 1, 0], [1, 3, 2]]])
    target = np.array([[[0, 1, 1], [0, 2, 2]], [[0, 1, 2], [2, 3, 1]]])

    TP, FP, FN, TN = confusion_matrix(input, target, threshold=1)
    print(TP, FP, FN, TN)

    TP_lxa, FN_lxa, FP_lxa, TN_lxa = prep_clf(input, target, threshold=1)
    print(TP_lxa, FN_lxa, FP_lxa, TN_lxa)

    print(precision(input, target, threshold=1))
    print(recall(input, target, threshold=1))
    print(ACC(input, target, threshold=1))
    print(FSC(input, target, threshold=1))
    print(HSS(input, target, threshold=1))
    print(BIAS(input, target, threshold=1))





