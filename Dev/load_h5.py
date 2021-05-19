import h5py
import sys
import matplotlib.pyplot as plt
import numpy as np

data_dir = 'C:/data2/no_padding_h5/3t4rRDI/'
data_dir2 = 'C:/data2/h5 file/new_low power/3t4r/RDI/'
filename = '3t4r_0_1_0_001.h5'
with h5py.File(data_dir + filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    b_group_key = list(f.keys())[1]

    # Get the data
    data = list(f[a_group_key])
    label = list(f[b_group_key])
    f.close()

frame1 = np.transpose(data[0], [2, 1, 0])
plt.figure(1)
plt.imshow(frame1[:, :, 0])
plt.show()

with h5py.File(data_dir2 + filename, "r") as f2:
    # List all groups
    print("Keys: %s" % f2.keys())
    a_group_key = list(f2.keys())[0]
    b_group_key = list(f2.keys())[1]

    # Get the data
    data = list(f2[a_group_key])
    label = list(f2[b_group_key])
    f2.close()

frame2 = np.transpose(data[0], [2, 1, 0])
plt.figure(2)
plt.imshow(frame2[:, :, 0])
plt.show()


# sys.exit()
