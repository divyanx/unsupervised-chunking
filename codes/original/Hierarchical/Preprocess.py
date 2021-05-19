import distance
import time
import numpy as np
import pandas as pd


def distance_matrix():
    dist = np.zeros(shape=(count+1, count+1))
    for i in range(0, count+1):
        for j in range(0, i):
            if i != j:
                dist[i][j] = ((data_points['x'][i] - data_points['x'][j])
                              ** 2 + (data_points['y'][i] - data_points['y'][j])**2)**0.5
                dist[j][i] = dist[i][j]
                values.insert(i, dist[i][j])
                print(i)
    values.sort()
    return dist


# MAIN
"""
Stores the distance matrix in a .npy file
"""
# f=open("data_amino2.txt","r").read()
# h=open("edited.txt","w")
# lines=f.splitlines()
data_points = pd.DataFrame({})
data_points['x'] = [22, 20, 28, 18, 29, 33, 34,
                    55, 45, 52, 51, 52, 55, 63, 55, 71, 64, 69, 72]
data_points['y'] = [39, 36, 30, 62, 54, 46, 55,
                    59, 63, 70, 76, 63, 58, 23, 14, 8, 29, 17, 34]

seq = dict()
values = list()
count = -1
count += len(data_points['x'])
start = time.time()
start = time.time()
a = distance_matrix()
print(a)
print(len(a), " ---- ", len(a[0]))
print("Distance Matrix Calculation done\t" + str(time.time()-start))
np.save('distance_matrix.npy', a)
