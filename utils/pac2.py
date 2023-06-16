import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 读取数据
# data = pd.read_csv('data.csv')
# X = data.iloc[:, :-1].values
# y = data.iloc[:, -1].values
factors_path = '../data/radar_npy/factors/100001_1.npy'
ob_path = '../data/radar_npy/ob/100001_1.npy'
factors = np.load(factors_path)
ob = np.load(ob_path)

X = factors.reshape(15, -1)
y = ob.reshape(1, -1)


# 执行主成分分析
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 绘制主成分分析图
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=np.ones(X_pca.shape[0]), cmap='viridis')
plt.colorbar()

# 添加标题和轴标签
plt.title("PCA Analysis of 15 Factors and Precipitation")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# plt.show()
plt.savefig('./pca/pca2.png')
