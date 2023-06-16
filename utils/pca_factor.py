import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt


def get_data():
    factors_path = '../data/radar_npy/factors/100001_1.npy'
    ob_path = '../data/radar_npy/ob/100001_1.npy'

    factors = np.load(factors_path)
    ob = np.load(ob_path)

    kind, h, w = factors.shape

    ob = ob.reshape(1, h, w)
    print(factors.shape, ob.shape)

    data = np.concatenate([factors, ob], axis=0)

    data = data.reshape(kind+1, -1)

    return data

def pca(data, n_components=3):
    pca = PCA(n_components=n_components)
    pca.fit(data)

    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

    return pca.transform(data)

def save_img(X_pca, path):
    print(X_pca.shape)
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c="blue", alpha=0.7)

    # 添加标题和轴标签
    ax.set_title("PCA Analysis of 15 Atmospheric Factors and Precipitation")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    # 添加图例
    ax.legend(["Atmospheric Factors and Precipitation"], loc="best")

    # plt.show()
    plt.savefig(path)

if __name__ == '__main__':
    data = get_data()
    X_pca = pca(data)
    save_img(X_pca, './pca/pca1.png')
