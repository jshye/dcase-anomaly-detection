import os
import sys
import glob
import numpy as np
import joblib
import scipy
from scipy.special import softmax
from sklearn import metrics
import torch


def calc_anomaly_score(preds, section_id):
    preds = softmax(preds.cpu(), axis=1)
    prob = preds[:, section_id]  # softmax output for the correct section
    anomaly_score = np.mean(
        np.log(
            np.maximum(1.0 - prob, sys.float_info.epsilon)
            - np.log(np.maximum(prob, sys.float_info.epsilon))
        ).numpy()
    )
    return anomaly_score



def fit_gamma_dist(anomaly_score, machine_type, epoch, config):
    gamma_params = scipy.stats.gamma.fit(anomaly_score)
    gamma_params = list(gamma_params)

    # fit gamma distribution for anomaly scores
    score_file_path = os.path.join(
        config['model_save_dir'], f'score_distr_{machine_type}_epoch{epoch}.pkl'
    )
    # save the parameters of the distribution
    joblib.dump(gamma_params, score_file_path)
    return gamma_params


def calc_decision_threshold(score_distr_file_path, config):
    # load anomaly score distribution for determining threshold
    shape_hat, loc_hat, scale_hat = joblib.load(score_distr_file_path)

    # determine threshold for decision
    decision_threshold = scipy.stats.gamma.ppf(
        q=config["decision_threshold"], a=shape_hat, loc=loc_hat, scale=scale_hat
    )

    return decision_threshold


def calc_evaluation_scores(y_true, y_pred, decision_threshold, config):
    try:
        auc = metrics.roc_auc_score(y_true, y_pred)
        p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=config["max_fpr"])
    except:
        auc = 0
        p_auc = 0

    _, false_positive, false_negative, true_positive = metrics.confusion_matrix(
        y_true, y_pred, labels=[True, False]).ravel()

    prec = true_positive / np.maximum(
        true_positive + false_positive, sys.float_info.epsilon
    )
    recall = true_positive / np.maximum(
        true_positive + false_negative, sys.float_info.epsilon
    )
    f1_score = 2.0 * prec * recall / np.maximum(prec + recall, sys.float_info.epsilon)

    print("AUC : {:.6f}".format(auc))
    print("pAUC : {:.6f}".format(p_auc))
    print("precision : {:.6f}".format(prec))
    print("recall : {:.6f}".format(recall))
    print("F1 score : {:.6f}".format(f1_score))

    return auc, p_auc, prec, recall, f1_score


def save_model(model, model_save_dir, machine_type):
    model_file_path = f'{model_save_dir}/model_{machine_type}.pt'

    torch.save(model.state_dict(), model_file_path)
    print("saved model -> %s" % (model_file_path))


def calc_performance_section(performance):
    """Calculate model performance per section"""
    csv_lines = []
    amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
    csv_lines.append(["arithmetic mean", ""] + list(amean_performance))
    hmean_performance = scipy.stats.hmean(
        np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon),
        axis=0,
    )
    csv_lines.append(["harmonic mean", ""] + list(hmean_performance))
    csv_lines.append([])

    return csv_lines


def calc_performance_all(performance):
    """Calculate model performance over all sections"""
    csv_lines = []
    csv_lines.append(["", "", "AUC", "pAUC", "precision", "recall", "F1 score"])
    amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
    csv_lines.append(
        ["arithmetic mean over all machine types, sections, and domains", ""]
        + list(amean_performance)
    )
    hmean_performance = scipy.stats.hmean(
        np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon),
        axis=0,
    )
    csv_lines.append(
        ["harmonic mean over all machine types, sections, and domains", ""]
        + list(hmean_performance)
    )
    csv_lines.append([])

    return csv_lines


class Visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(7, 5))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)


import matplotlib.pyplot as plt

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