import numpy as np
import matplotlib.pyplot as plt
import torch


def save_model(model, model_save_dir, machine_type):
    model_file_path = f'{model_save_dir}/model_{machine_type}.pt'

    torch.save(model.state_dict(), model_file_path)
    print("saved model -> %s" % (model_file_path))


def visualize(loss, val_loss, fname):
    fig = plt.figure(figsize=(7,5))
    plt.plot(loss, 'o-', label='train', markersize=3)
    plt.plot(val_loss, 'x-', label='valid', markersize=3)
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def plot_anomaly_score_distrib(normal, abnormal, epoch, decimals=2, show=False):
    normal_scores = np.around(normal, decimals=decimals)
    unique, counts = np.unique(normal_scores, return_counts=True)
    normal_scores_cnt = dict(zip(unique, counts))

    abnormal_scores = np.around(abnormal, decimals=decimals)
    unique, counts = np.unique(abnormal_scores, return_counts=True)
    abnormal_scores_cnt = dict(zip(unique, counts))
    
    fig = plt.figure(figsize=(7,5))
    plt.bar(*zip(*normal_scores_cnt.items()), label='normal', width=0.05, edgecolor='k', alpha=0.2)
    plt.bar(*zip(*abnormal_scores_cnt.items()), label='anomaly', width=0.05, edgecolor='k', alpha=0.2)
    plt.title(f'[{epoch}] Anomaly Score Distribution')
    plt.legend()
    plt.tight_layout()

    if show:
        plt.show()
    else:
        return fig


def plot_roc(fpr, tpr, auc, show=False):
    fig = plt.figure(figsize=(4,4))
    plt.plot(fpr, tpr)
    plt.title(f'ROC-AUC: {auc:.4f}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.tight_layout()

    if show:
        plt.show()
    else:
        return fig


def plot_probability(label_list, anomaly_prob, idx, anomaly=True, show=False):
    fig = plt.figure(figsize=(5,4))
    x_list = np.arange(len(label_list))
    plt.bar(x_list, anomaly_prob)
    plt.xticks(x_list, label_list)
    plt.ylim([0, 1])
    plt.grid()

    if anomaly:
        plt.title(f'(Anomalous {idx}) Predicted Probability')
    else:
        plt.title(f'(Normal {idx}) Predicted Probability')

    if show:
        plt.show()
    else:
        return fig