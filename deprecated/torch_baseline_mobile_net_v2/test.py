"""PyTorch script for test (MobileNetV2).

Copyright (C) 2021 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# Standard library imports.
import os
import sys

import joblib

# Related third party imports.
import numpy as np
import scipy.stats
import torch
import torch.utils.data
from scipy.special import softmax
from sklearn import metrics

# Local application/library specific imports.
import util
from pytorch_model import MobileNetV2

# Load configuration from YAML file.
CONFIG = util.load_yaml("./config.yaml")

# String constant: "cuda:0" or "cpu"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(machine_type, n_sections):
    """
    Load model file
    """
    model_file = "{model}/model_{machine_type}.hdf5".format(
        model=CONFIG["model_directory"], machine_type=machine_type
    )
    if not os.path.exists(model_file):
        print("{} model not found ".format(machine_type))
        sys.exit(-1)

    model = MobileNetV2(n_sections).to(DEVICE)
    model.eval()
    model.load_state_dict(torch.load(model_file))

    return model


def calc_decision_threshold(target_dir):
    """
    Calculate decision_threshold from anomaly score distribution.
    """

    # load anomaly score distribution for determining threshold
    score_distr_file_path = "{model}/score_distr_{machine_type}.pkl".format(
        model=CONFIG["model_directory"], machine_type=os.path.split(target_dir)[1]
    )
    shape_hat, loc_hat, scale_hat = joblib.load(score_distr_file_path)
    # determine threshold for decision
    decision_threshold = scipy.stats.gamma.ppf(
        q=CONFIG["decision_threshold"], a=shape_hat, loc=loc_hat, scale=scale_hat
    )

    return decision_threshold


def calc_anomaly_score(model, file_path, section_index):
    """
    Calculate anomaly score.
    """
    try:
        # extract features (log-mel spectrogram)
        data = util.extract_feature(file_path, config=CONFIG["feature"])
        data = data.reshape(
            data.shape[0],
            1,
            CONFIG["feature"]["n_frames"],
            CONFIG["feature"]["n_mels"],
        )
    except FileNotFoundError:
        print("File broken!!: {}".format(file_path))

    condition = np.zeros((data.shape[0]), dtype=int)
    if section_index != -1:
        condition[:] = section_index

    feed_data = torch.from_numpy(data).clone()
    feed_data = feed_data.to(DEVICE).float()
    with torch.no_grad():
        output = model(feed_data)  # notice: unnormalized output
        output = output.to("cpu").detach().numpy().copy()  # tensor to numpy array.

    output = softmax(output, axis=1)
    prob = output[:, section_index]

    y_pred = np.mean(
        np.log(
            np.maximum(1.0 - prob, sys.float_info.epsilon)
            - np.log(np.maximum(prob, sys.float_info.epsilon))
        )
    )

    return y_pred


def test_section(model, test_files, decision_threshold, score_list, section_index):
    """
    Test a section (almost equal to machine id).
    """
    # section_idx = section_info[1]

    # setup anomaly score file path
    anomaly_score_list = []

    # setup decision result file path
    decision_result_list = []

    y_pred = [0.0 for k in test_files]
    for file_idx, file_path in enumerate(test_files):
        y_pred[file_idx] = calc_anomaly_score(
            model, file_path=file_path, section_index=section_index
        )
        anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])

        # store decision results
        if y_pred[file_idx] > decision_threshold:
            decision_result_list.append([os.path.basename(file_path), 1])
        else:
            decision_result_list.append([os.path.basename(file_path), 0])

    score_list["anomaly"] = anomaly_score_list
    score_list["decision"] = decision_result_list

    return y_pred


def save_anomaly_score(score_list, target_dir, section_name, dir_name):
    """
    Save anomaly scores and decision results.

    score_list : anomaly scores and decision results (type: dictionary).
    """

    # output anomaly scores
    util.save_csv(
        save_file_path="{result}/anomaly_score_{machine_type}"
        "_{section_name}_{dir_name}.csv".format(
            result=CONFIG["result_directory"],
            machine_type=os.path.split(target_dir)[1],
            section_name=section_name,
            dir_name=dir_name,
        ),
        save_data=score_list["anomaly"],
    )

    # output decision results
    util.save_csv(
        save_file_path="{result}/decision_result_{machine_type}"
        "_{section_name}_{dir_name}.csv".format(
            result=CONFIG["result_directory"],
            machine_type=os.path.split(target_dir)[1],
            section_name=section_name,
            dir_name=dir_name,
        ),
        save_data=score_list["decision"],
    )


def calc_evaluation_scores(y_true, y_pred, decision_threshold):
    """
    Calculate evaluation scores (AUC, pAUC, precision, recall, and F1 score)
    """
    auc = metrics.roc_auc_score(y_true, y_pred)
    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=CONFIG["max_fpr"])

    (_, false_positive, false_negative, true_positive,) = metrics.confusion_matrix(
        y_true, [1 if x > decision_threshold else 0 for x in y_pred]
    ).ravel()

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


def calc_performance_section(performance, csv_lines):
    """
    Calculate model performance per section.
    """
    amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
    csv_lines.append(["arithmetic mean", ""] + list(amean_performance))
    hmean_performance = scipy.stats.hmean(
        np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon),
        axis=0,
    )
    csv_lines.append(["harmonic mean", ""] + list(hmean_performance))
    csv_lines.append([])

    return csv_lines


def calc_performance_all(performance, csv_lines):
    """
    Calculate model performance over all sections.
    """
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


def save_result(csv_lines):
    """
    Save averages for AUCs and pAUCs.
    """

    result_path = "{result}/{file_name}".format(
        result=CONFIG["result_directory"], file_name=CONFIG["result_file"]
    )
    print("results -> {}".format(result_path))
    util.save_csv(save_file_path=result_path, save_data=csv_lines)


def main():
    """
    Perform model evaluation.
    """

    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = util.command_line_chk()  # constant: True or False
    if mode is None:
        sys.exit(-1)

    # make result directory
    os.makedirs(CONFIG["result_directory"], exist_ok=True)

    # load base_directory list
    dir_list = util.select_dirs(config=CONFIG, mode=mode)

    # initialize lines in csv for AUC and pAUC
    csv_lines = []
    performance = {"section": None, "all": None}

    # anomaly scores and decision results
    score_list = {"anomaly": None, "decision": None}

    if mode:
        performance["all"] = []

    for idx, target_dir in enumerate(dir_list):
        print("===============================================")
        print("[%d/%d] %s" % (idx + 1, len(dir_list), target_dir))

        print("================ MODEL LOAD =================")

        machine_type = os.path.split(target_dir)[1]
        section_names_file_path = "{model}/section_names_{machine_type}.pkl".format(
            model=CONFIG["model_directory"], machine_type=machine_type
        )
        trained_section_names = joblib.load(section_names_file_path)
        n_sections = trained_section_names.shape[0]

        model = load_model(
            machine_type=os.path.split(target_dir)[1], n_sections=n_sections
        )
        decision_threshold = calc_decision_threshold(target_dir)

        if mode:
            # results for each machine type
            csv_lines.append([os.path.split(target_dir)[1]])  # append machine type
            csv_lines.append(
                ["section", "domain", "AUC", "pAUC", "precision", "recall", "F1 score"]
            )
            performance["section"] = []

        for dir_name in ["source_test", "target_test"]:
            for section_name in util.get_section_names(target_dir, dir_name=dir_name):

                # search for section_name
                temp_array = np.nonzero(trained_section_names == section_name)[0]
                if temp_array.shape[0] == 0:
                    section_idx = -1
                else:
                    section_idx = temp_array[0]

                # load test file
                test_files, y_true = util.file_list_generator(
                    target_dir=target_dir,
                    section_name=section_name,
                    dir_name=dir_name,
                    mode=mode,
                )

                print(
                    "============== BEGIN TEST FOR A SECTION %s OF %s =============="
                    % (section_name, dir_name)
                )
                # - perform test for a section
                # - anomaly scores and decision results are saved in score_list
                y_pred = test_section(
                    model,
                    test_files,
                    decision_threshold,
                    score_list,
                    section_idx,
                )

                # save anomaly scores and decision results
                save_anomaly_score(score_list, target_dir, section_name, dir_name)

                if mode:
                    # evaluation_scores (list): auc, p_auc, prec, recall, f1_score
                    eval_scores = calc_evaluation_scores(
                        y_true, y_pred, decision_threshold
                    )
                    csv_lines.append(
                        [
                            section_name.split("_", 1)[1],
                            dir_name.split("_", 1)[0],
                            *eval_scores,  # unpack
                        ]
                    )
                    performance["section"].append(eval_scores)
                    performance["all"].append(eval_scores)

                print(
                    "============ END OF TEST FOR A SECTION %s OF %s ============\n"
                    % (section_name, dir_name)
                )

        if mode:
            # calculate averages for AUCs and pAUCs
            csv_lines = calc_performance_section(performance["section"], csv_lines)

        del model

    if mode:
        # calculate averages for AUCs and pAUCs over all sections
        csv_lines = calc_performance_all(performance["all"], csv_lines)

        # output results
        save_result(csv_lines)


if __name__ == "__main__":
    main()
