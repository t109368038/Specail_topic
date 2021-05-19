import numpy as np

data  = np.load("D:/kaiku_report/2021-0418for_posheng/adata.npy")

data = np.reshape(data, [-1, 4])
data = data[:, 0:2:] + 1j * data[:, 2::]
data = np.reshape(data, [-1,16, 12, 64])
print(np.shape(data))