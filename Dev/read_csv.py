import csv
import numpy as np
import matplotlib.pyplot as plt

path = 'E:/ResearchData/'
data = []
with open(path + '1.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        # print(row)
        data.append(row)

data = np.array(data)
plt.figure()

