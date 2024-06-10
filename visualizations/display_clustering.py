import matplotlib.pyplot as plt


def display_clustering(dimension, labels, data, cluster_num):
    if dimension <= 2:
        plt.figure(figsize=(8, 8))
        color = ['red', 'aqua', 'yellow', 'magenta', 'black', 'darkgreen', 'blue', 'gray', 'purple', 'lime', 'maroon',
                 'gold', 'deeppink', 'silver', 'lawngreen', 'pink', 'navy', 'blueviolet', 'turquoise', 'dodgerblue',
                 'navajowhite', 'khaki', 'darkslateblue', 'darkseagreen', 'mediumvioletred', 'palevioletred',
                 'cornflowerblue', 'plum', 'steelblue', 'lightcoral']
        if cluster_num <= len(color):
            for i in range(len(data)):
                if labels[i] != -1:
                    plt.scatter(data[i][0], data[i][1], c=color[labels[i] - 1])
                else:
                    plt.scatter(data[i][0], data[i][1], c='green', marker='*', s=150)
        else:
            for i in range(len(data)):
                if labels[i] == -1:
                    plt.scatter(data[i][0], data[i][1], c='green', marker='*', s=150)
                else:
                    plt.scatter(data[i][0], data[i][1], c='blue', label='Datapoints')
        plt.xlabel('X-Coordinates')
        plt.ylabel('Y-Coordinates')
        plt.title('Datapoints After Outlier Filtration')
        plt.show()
