import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import DSP_2t4r

data_folder = '../data/adc_data_3t4r_np_format/'
data = np.load(data_folder + 'np_3t4r0_0_8_005.npy')

# data_folder_0 = '../data/Studio_data/2t4r_v132/'
# data_folder_1 = '../data/Studio_data/2t4r_v123/'
# data_folder_2 = '../data/Studio_data/2t4r_v3/'
# filename = 'RAI_S0G0.npy'

# data0 = np.load(data_folder_0 + filename)
# data1 = np.load(data_folder_1 + filename)
# data2 = np.load(data_folder_2 + filename)

colors = [[62, 38, 168, 255], [63, 42, 180, 255], [65, 46, 191, 255], [67, 50, 202, 255], [69, 55, 213, 255],
          [70, 60, 222, 255], [71, 65, 229, 255], [70, 71, 233, 255], [70, 77, 236, 255], [69, 82, 240, 255],
          [68, 88, 243, 255],
          [68, 94, 247, 255], [67, 99, 250, 255], [66, 105, 254, 255], [62, 111, 254, 255], [56, 117, 254, 255],
          [50, 123, 252, 255],
          [47, 129, 250, 255], [46, 135, 246, 255], [45, 140, 243, 255], [43, 146, 238, 255], [39, 150, 235, 255],
          [37, 155, 232, 255],
          [35, 160, 229, 255], [31, 164, 225, 255], [24, 173, 219, 255], [17, 177, 214, 255],
          [7, 181, 208, 255],
          [1, 184, 202, 255], [2, 186, 195, 255], [11, 189, 188, 255], [24, 191, 182, 255], [36, 193, 174, 255],
          [44, 195, 167, 255],
          [49, 198, 159, 255], [55, 200, 151, 255], [63, 202, 142, 255], [74, 203, 132, 255], [88, 202, 121, 255],
          [102, 202, 111, 255],
          [116, 201, 100, 255], [130, 200, 89, 255], [144, 200, 78, 255], [157, 199, 68, 255], [171, 199, 57, 255],
          [185, 196, 49, 255],
          [197, 194, 42, 255], [209, 191, 39, 255], [220, 189, 41, 255], [230, 187, 45, 255], [239, 186, 53, 255],
          [248, 186, 61, 255],
          [254, 189, 60, 255], [252, 196, 57, 255], [251, 202, 53, 255], [249, 208, 50, 255], [248, 214, 46, 255],
          [246, 220, 43, 255],
          [245, 227, 39, 255], [246, 233, 35, 255], [246, 239, 31, 255], [247, 245, 27, 255], [249, 251, 20, 255]]
colors = np.array(colors) / 256
new_cmp = ListedColormap(colors)

f = 30
# data = np.concatenate([data[f, :, :, 0:4], data[f, :, :, 8:12]])
data = data[f, :, :, 0:4]
rai = DSP_2t4r.Range_Angle(data, mode=1, padding_size=[128, 64, 32])
plt.figure()
plt.imshow(rai.sum(0), cmap=new_cmp)



# plt.figure(1)
# plt.imshow(data0[63, 29, :, :], new_cmp)

# plt.figure(2)
# plt.imshow(data1[63, 29, :, :], new_cmp)

# plt.figure(3)
# plt.imshow(data2[63, 29, :, :], new_cmp)
# plt.show()