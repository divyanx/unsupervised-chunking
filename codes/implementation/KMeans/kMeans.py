import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from random import randint


class KMeans:

    def __init__(self):

        # Number of clusters
        self.k = 0

        # Load Data
        self.load_data()

        # Initializing Centroids
        self.min_x, self.max_x = self.data_points['x'].min(), self.data_points['x'].max()
        self.min_y, self.max_y = self.data_points['y'].min(), self.data_points['y'].max()
        self.centroids = { }

        for i in range(1, self.k+1):
            self.centroids[i] = [
                np.random.randint(self.min_x, self.max_x),
                np.random.randint(self.min_y, self.max_y)
            ]

        # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        # self.colors_for_plot = {}
        # for i, color in enumerate(colors):
            # self.colors_for_plot[i+1] = color


    def load_data(self):

        if len(sys.argv) <= 2:
            print("No data file(s) provided!")
            exit(-1)

        read_lines = []
        with open(sys.argv[1], "r") as inpFile:
            read_lines = inpFile.readlines()

        data_words = []
        sentences_added = 0
        line_num = 0
        while sentences_added < max_sentences:
            if len(read_lines[line_num]) > 1:
                data_words.append(read_lines[line_num])
            else:
                sentences_added += 1
            line_num += 1

        self.word_tags = [_.split("\t") for _ in data_words]


        chunk_types = [
            ["JJ", "N","PSP", "QT", "RD_SYM", "DM"],
            ["PR"],
            ["RD_PUNC"],
            ["CC"],
            ["V"],
            ["RD_UNK"]
        ]

        X_coordinates = []
        Y_coordinates = []

        for word_num, word_tag in enumerate(self.word_tags):
            X_coordinates.append(scale_factor*word_num)
            Y_coord = 0
            match_found = False
            for chunk_num, chunk_type in enumerate(chunk_types):
                for tag in chunk_type:
                    if word_tag[1].startswith(tag):
                        if(tag == "N"):
                            self.k += 1
                        match_found = True
                        Y_coord = chunk_num
                        break
                if match_found:
                    break
            if not match_found:
                Y_coord = randint(0, 5)

            Y_coordinates.append(Y_coord)

        self.data_points = pd.DataFrame({})
        self.data_points['x'] = X_coordinates
        self.data_points['y'] = Y_coordinates


    def assign_centroids(self):
        for i in self.centroids.keys():
            self.data_points[f'distance_with_{i}th'] = KMeans.get_distance(
                self.data_points['x'], self.data_points['y'],
                self.centroids[i][0], self.centroids[i][1]
            )


        centroids_distance_cols = [f'distance_with_{i}th' for i in self.centroids.keys()]
        self.data_points['nearest'] = self.data_points.loc[:, centroids_distance_cols].idxmin(axis=1)
        self.data_points['nearest'] = self.data_points['nearest'].map(lambda x: int(x.lstrip('distance_with_')[:-2]))
        # self.data_points['color'] = self.data_points['nearest'].map(lambda x: self.colors_for_plot[x])


    def update_centroids(self):
        for i in self.centroids.keys():
            self.centroids[i][0] = np.mean(self.data_points[self.data_points['nearest'] == i]['x'])
            self.centroids[i][1] = np.mean(self.data_points[self.data_points['nearest'] == i]['y'])


    def train_model(self):

        self.assign_centroids()
        self.update_centroids()

        has_converged = False

        while not has_converged:
            nearest_centroids = self.data_points['nearest'].copy(deep=True)
            self.update_centroids()
            self.assign_centroids()

            if nearest_centroids.equals(self.data_points['nearest']):
                has_converged = True

    def plot_centroids(self):

        for centroid in self.centroids.keys():
            print(self.centroids[centroid])

        fig = plt.figure(figsize=(5,5))
        plt.scatter(self.data_points['x'], self.data_points['y']
                    # color=self.data_points['color'], alpha=0.5, edgecolor='k'
                    )

        for i in self.centroids.keys():
            plt.scatter(*self.centroids[i]
                        # color=self.colors_for_plot[i]
                        )
        plt.xlim(self.min_x - 10, self.max_x + 10)
        plt.xlim(self.min_y - 10, self.max_y + 10)
        plt.show()

    def show_stats(self):

        output_fp = open(sys.argv[2], "w")
        last_centroid = -1
        chunk_type = "S"
        for point_num in range(len(self.data_points["x"])):
            point = self.data_points["x"][point_num]
            list_num = self.data_points["y"][point_num]
            current_centroid = self.data_points["nearest"][point_num]
            if current_centroid != last_centroid:
                last_centroid = current_centroid
                if list_num == 0:
                    chunk_type = "NP"
                elif list_num == 1:
                    chunk_type = "PR"
                elif list_num == 2:
                    chunk_type = "RD_PUNC"
                elif list_num == 3:
                    chunk_type = "CC"
                elif list_num == 4:
                    chunk_type = "V"
                elif list_num == 5:
                    chunk_type = "RD_UNK"
                print(self.word_tags[int(round(point/scale_factor))][0], "\tB-"+chunk_type, file=output_fp)
            else:
                print(self.word_tags[int(round(point/scale_factor))][0], "\tI-"+chunk_type, file=output_fp)

            if self.word_tags[int(round(point/scale_factor))][0] == "।":
                print("", file=output_fp)

    @staticmethod
    def get_distance(x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return np.sqrt( (dx ** 2) + (dy ** 2) )


if __name__ == "__main__":

    max_sentences = 10
    scale_factor = 1.4


    np.random.seed(100)

    # k = int(input("Enter the number of centroids (k): "))

    kmeans = KMeans()

    kmeans.train_model()

    kmeans.show_stats()
