import matplotlib.pyplot as plt
import numpy as np


class ROCLogger:
    def __init__(self, title):
        self.title = title
        self.fig = plt.figure(figsize=(8,4))

    def plot_roc(self, fpr, tpr, normal, abnormal):
        self.fig.add_subplot(1,2,1)
        plt.plot(fpr, tpr)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.title(self.title)

        self.fig.add_subplot(1,2,2)
        normal_scores = np.around(normal, decimals=2)
        unique, counts = np.unique(normal_scores, return_counts=True)
        normal_scores_cnt = dict(zip(unique, counts))

        abnormal_scores = np.around(abnormal, decimals=2)
        unique, counts = np.unique(abnormal_scores, return_counts=True)
        abnormal_scores_cnt = dict(zip(unique, counts))
        
        plt.bar(*zip(*normal_scores_cnt.items()), label='normal', width=0.02, edgecolor='k', alpha=0.2)
        plt.bar(*zip(*abnormal_scores_cnt.items()), label='anomaly', width=0.02, edgecolor='k', alpha=0.2)
        plt.xlabel('Anomaly Score')
        plt.ylabel('Counts')
        plt.legend()
        plt.tight_layout()

    def save_fig(self, save_path):
        plt.savefig(save_path)
        plt.close()


class TrainLogger:
    def __init__(self, title, xlabel, ylabel, i):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fignum = i
        self.fig = plt.figure(num=self.fignum, figsize=(7,5))
    
    def plot_learning_curve(self, values, label):
        plt.figure(self.fignum)
        plt.plot(np.arange(len(values)), values, label=label, marker='o', markersize=2)

    def save_fig(self, save_path):
        plt.figure(self.fignum)
        plt.title(self.title)
        plt.legend()
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.grid(alpha=0.2)
        plt.savefig(save_path)
        plt.close(self.fignum)
