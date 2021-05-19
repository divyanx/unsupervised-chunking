import distance
import time
import numpy as np
import pandas as pd
import sys
from random import randint


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


"""
Stores the distance matrix in a .npy file
"""
# f=open("data_amino2.txt","r").read()
# h=open("edited.txt","w")
# lines=f.splitlines()

if len(sys.argv) == 1:
    print("No data file provided!")
    exit(-1)

read_lines = []
with open(sys.argv[1], "r") as inpFile:
    read_lines = inpFile.readlines()

data_words = [_.replace("\n", "") for _ in read_lines if len(_) > 1]
data_words = []
sentences_added = 0
line_num = 0
while sentences_added < 50:
    if len(read_lines[line_num]) > 1:
        data_words.append(read_lines[line_num])
    else:
        sentences_added += 1
    line_num += 1

word_tags = [_.split("\t") for _ in data_words]


chunk_types = [
    ["JJ", "N", "PSP", "QT", "RD_SYM", "DM"],
    ["PR"],
    ["RD_PUNC"],
    ["CC"],
    ["V"],
    ["RD_UNK"]
]
val1 = 0
val2 = 100
val3 = 200
val4 = 300
val5 = 400
val5 = 500
val6 = 600
chunk_type_dict = {
    "J":  val1,
    "N": val1,
    "P": val1,
    "Q": val1,
    "RD_SYM": val1,
    "D": val1,
    "P": val2,
    "RD_PUNC": val3,
    "C": val4,
    "V": val5,
    "RD_UNK": val6
}
x_cordinates = []
y_cordinates = []
words = []
word_index_dict = {}
word_occurance_count = []
word_location = []
word_type_val = []

snt_index = 0
index_in_list = 0

for word, w_type in word_tags:
    if word not in word_index_dict.keys():
        word_index_dict[word] = index_in_list
        word_occurance_count.append(1)
        word_location.append(snt_index)
        if w_type[0] != 'R':
            word_type_val.append(chunk_type_dict[w_type[0]])
        elif w_type in chunk_type_dict.keys():
            word_type_val.append(chunk_type_dict[w_type])
        else:
            word_type_val.append(700)
        words.append(word)
        index_in_list += 1
    else:
        print("index is ", word_index_dict[word])
        print("length is",  len(word_occurance_count))
        word_occurance_count[word_index_dict[word]] += 1
        word_location[word_index_dict[word]] += snt_index
        if w_type[0] != 'R':
            word_type_val[word_index_dict[word]] = chunk_type_dict[w_type[0]]
        elif w_type in chunk_type_dict.keys():
            word_type_val[word_index_dict[word]] = chunk_type_dict[w_type]
        else:
            word_type_val[word_index_dict[word]] = 700
    if w_type == "RD_PUNC":
        snt_index = 0
    else:
        snt_index += 1


for i in range(len(word_occurance_count)):
    x_cordinates.append(word_type_val[i])
    y_cordinates.append(250 * word_location[i]/word_occurance_count[i])

data_points = pd.DataFrame({})
data_points['x'] = x_cordinates
data_points['y'] = y_cordinates

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
