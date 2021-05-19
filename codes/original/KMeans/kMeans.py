import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class KMeans:

    def __init__(self, k):

        # Number of clusters
        self.k = k

        # Load Data
        self.load_data()

        # Initializing Centroids
        self.min_x, self.max_x = self.data_points['x'].min(), self.data_points['x'].max()
        self.min_y, self.max_y = self.data_points['y'].min(), self.data_points['y'].max()
        self.centroids = { }

        for i in range(1, k+1):
            self.centroids[i] = [
                np.random.randint(self.min_x, self.max_x),
                np.random.randint(self.min_y, self.max_y)
            ]

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        self.colors_for_plot = {}
        for i, color in enumerate(colors):
            self.colors_for_plot[i+1] = color


    def load_data(self):
        self.data_points = pd.DataFrame({})
        self.data_points['x'] = [22, 20, 28, 18, 29, 33, 34, 55, 45, 52, 51, 52, 55, 63, 55, 71, 64, 69, 72]
        self.data_points['y'] = [39, 36, 30, 62, 54, 46, 55, 59, 63, 70, 76, 63, 58, 23, 14, 8, 29, 17, 34]


    def assign_centroids(self):
        for i in self.centroids.keys():
            self.data_points[f'distance_with_{i}th'] = KMeans.get_distance(
                self.data_points['x'], self.data_points['y'],
                self.centroids[i][0], self.centroids[i][1]
            )


        centroids_distance_cols = [f'distance_with_{i}th' for i in self.centroids.keys()]
        self.data_points['nearest'] = self.data_points.loc[:, centroids_distance_cols].idxmin(axis=1)
        self.data_points['nearest'] = self.data_points['nearest'].map(lambda x: int(x.lstrip('distance_with_')[:-2]))
        self.data_points['color'] = self.data_points['nearest'].map(lambda x: self.colors_for_plot[x])


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
        plt.scatter(self.data_points['x'], self.data_points['y'], color=self.data_points['color'], alpha=0.5, edgecolor='k')

        for i in self.centroids.keys():
            plt.scatter(*self.centroids[i], color=self.colors_for_plot[i])
        plt.xlim(self.min_x - 10, self.max_x + 10)
        plt.xlim(self.min_y - 10, self.max_y + 10)
        plt.show()


    @staticmethod
    def get_distance(x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return np.sqrt( (dx ** 2) + (dy ** 2) )


if __name__ == "__main__":

    np.random.seed(100)

    k = int(input("Enter the number of centroids (k): "))

    kmeans = KMeans(k)

    kmeans.train_model()

    kmeans.plot_centroids()
