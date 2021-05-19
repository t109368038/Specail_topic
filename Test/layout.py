import  numpy as np
x = [1,2,3,4,5,6,7,8,9,10,11,12]
print(type(x))
data=np.array(x)
data = np.reshape(data, [-1, 4])
print(data)
# # data = data[:, 0:2:] + 1j * data[:, 2::]
data = data[:, 0:2:] + 1j * data[:, 2::]
print(data)
