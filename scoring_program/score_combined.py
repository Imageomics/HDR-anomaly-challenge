#!/usr/bin/env python

# Scoring program for the HDR Anomaly Challenge

import os
import pathlib
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score, roc_auc_score, accuracy_score

# Constants
HELPER_DIRECTORY = str(pathlib.Path(__file__).parent.resolve() / "helper_scripts")

# Importing functions from helper_scripts
sys.path.append(HELPER_DIRECTORY)
from dataio import parse_prediction_file, parse_solution_file, save_scores

'''
# This is for local testing only, along with DEFAULT_CONFIG defined above
def get_config():
    config_path = DEFAULT_CONFIG
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        
    return parse_yaml_file(config_path)
'''

# def evaluate_prediction(score_df):
#     score_df["preds"], score_df["hybrid_stat"]

#     score_df['threshold_preds'] = score_df

#     threshold_preds = None
#     for i in range(combined.shape[0] + 1):
#         ls = len(combined[:i])
#         rs = len(combined[i:])
#         preds = np.concatenate((np.zeros(ls), np.ones(rs)))
#         recall = recall_score(combined[:, 1], preds, pos_label=0)
#         if recall >= 0.95:
#             threshold_i = i
#             threshold_preds = preds
#             break
    
#     return threshold_preds, combined[:, 1]

def evaluate_prediction(score_df):
    # loop predictions from most likely non-hybrid to most likeyly hybrid
    for threshold_pred in sorted(set(score_df["preds"])):
        score_df['converted_preds'] = score_df["preds"].apply(lambda x: 1 if x > threshold_pred else 0)
        threshold_recall = recall_score(score_df["hybrid_stat"], score_df["converted_preds"], pos_label=0)  # non-hybrid is the positive here, so positive label is 0
        if threshold_recall >= 0.95:
            break
    
    print(f'With non-hybrid recall {str(round(threshold_recall, 4))}, the predictions equal and lower than the threshold confident score {str(threshold_pred)} are all non-hybrids and the ones higher are all hyrids.')

    return score_df, threshold_recall, threshold_pred

    # combined = list(zip(scores, labels))
    # # combined = sorted(combined, key=lambda x: x[0], reverse=reversed)
    # combined = np.array(combined).astype(np.float32)
    # threshold_i = None
    # threshold_preds = None
    # for i in range(combined.shape[0] + 1):
    #     ls = len(combined[:i])
    #     rs = len(combined[i:])
    #     preds = np.concatenate((np.zeros(ls), np.ones(rs)))
    #     recall = recall_score(combined[:, 1], preds, pos_label=0)
    #     if recall >= 0.95:
    #         threshold_i = i
    #         threshold_preds = preds
    #         break
    
    # return threshold_preds, combined[:, 1]

# def evaluate(preds, gt):
#     """Requires lower score to mean more likely to be non-hybrid,
#     and higher score to mean more likely to be hybrid.
    
#     If you would like this to be reversed, set reversed=True
#     """
#     h_recall = recall_score(gt, preds, pos_label=1)
#     h_precision = precision_score(gt, preds, pos_label=1)
#     f1 = f1_score(gt, preds, pos_label=1)
#     roc_auc = roc_auc_score(gt, preds)
#     acc = accuracy_score(gt, preds)

#     return h_recall, h_precision, f1, roc_auc, acc

def evaluate_major_minor_prediction(score_df):
    print("Evaluating performance on signal vs non-signal hybrids")
    # Set to compare just hybrids and look if they're predicted as such
    major_true_df = score_df.loc[(score_df["ssp_indicator"] == "major") & (score_df["hybrid_stat"] == 1)].copy()
    minor_true_df = score_df.loc[(score_df["ssp_indicator"] == "minor") & (score_df["hybrid_stat"] == 1)].copy()

    major_recall = recall_score(major_true_df["hybrid_stat"], major_true_df["converted_preds"])
    minor_recall = recall_score(minor_true_df["hybrid_stat"], minor_true_df["converted_preds"])

    major_true_df = score_df.loc[score_df["ssp_indicator"] == "major"].copy()
    minor_true_df = score_df.loc[score_df["ssp_indicator"] == "minor"].copy()

    major_roc_auc = roc_auc_score(major_true_df["hybrid_stat"], major_true_df["preds"])
    minor_roc_auc = roc_auc_score(minor_true_df["hybrid_stat"], minor_true_df["preds"])

    major_prc_auc = average_precision_score(major_true_df["hybrid_stat"], major_true_df["preds"])
    minor_prc_auc = average_precision_score(minor_true_df["hybrid_stat"], minor_true_df["preds"])
    
    scores = {
        "major_recall" : major_recall,
        "minor_recall" : minor_recall,
        "major_prc_auc" : major_prc_auc,
        "minor_prc_auc" : minor_prc_auc,
        "major_roc_auc" : major_roc_auc,
        "minor_roc_auc" : minor_roc_auc,
    }
    # print("signal, non-signal hybrid detection scores: ", scores)
    
    return scores

def score_predictions(score_df, mm_vals=False):
    
    # preds, gt = evaluate_prediction(score_df["preds"], score_df["hybrid_stat"])
    # score_df['converted_preds'] = preds

    score_df, threshold_recall, threshold_pred = evaluate_prediction(score_df)
    gt = score_df["hybrid_stat"]
    preds = score_df["converted_preds"]

    # metrics for hybrids, hybrids are positive here
    h_recall = recall_score(gt, preds, pos_label=1)
    h_precision = precision_score(gt, preds, pos_label=1)
    f1 = f1_score(gt, preds, pos_label=1)
    acc = accuracy_score(gt, preds)
    prc_auc = average_precision_score(gt, score_df["preds"], pos_label=1) #better when imbalanced class, focus on positive rare; average_precision_score approximates the AUC by a sum over the precisions at every possible threshold value, better than Trapezoidal Rule when the curve has significant non-linearities
    roc_auc = roc_auc_score(gt, score_df["preds"])
    
    # h_recall, h_precision, f1, roc_auc, acc = evaluate(preds, gt)
    
    scores = {
        "threshold_recall" : float(round(threshold_recall, 4)),
        "threshold_pred" : float(threshold_pred),
        "hybrid_recall" : float(h_recall),
        "hybrid_precision" : float(h_precision),
        "f1_score" : float(f1),
        "accuracy" : float(acc),
        "prc_auc" : float(prc_auc),
        "roc_auc" : float(roc_auc)
    }
    
    if mm_vals:
        mm_scores = evaluate_major_minor_prediction(score_df)
        #mm_scores = evaluate_major_minor_prediction(pred_vals, sol_gt_aligned, mm_vals_aligned, score_df, reversed=reverse_score_prediction)
        scores.update(mm_scores)
        print(f"Full Scores Species A hybrid detection: {scores}")
    else:
        print(f"Full Scores Mimic hybrid detection: {scores}")
        
    return scores


def get_scores(pred_df=None, sol_df=None, mm_vals = False):
    # merge ref with predictions
    # aligns the ref values with scores in the columns based on filenames
    score_df = pd.merge(sol_df, pred_df, on = "filename", how = "inner")
    # score_df = score_df.sort_values(by='preds')
    
    # Check aligned on all expected files
    if score_df.shape[0] != sol_df.shape[0]:
        sys.exit(f"There should have been {sol_df.shape[0]} predictions, but we only got {score_df.shape[0]}")
    
    scores = score_predictions(score_df, mm_vals) #, config.reverse_score_prediction)
    
    return scores


if __name__ == "__main__":
    # Get scoring configurations -- for local testing
    #config = get_config()
    
    # Directory to read labels from
    input_dir = sys.argv[1]
    solutions = os.path.join(input_dir, 'ref')
    prediction_dir = os.path.join(input_dir, 'res')


    print("Using input_dir: " + input_dir)
    print("Using solutions: " + solutions)
    print("Using prediction_dir: " + prediction_dir)

    # Directory to output computed score into
    output_dir = sys.argv[2]
    print("Using output_dir: " + output_dir)

    prediction_file = os.path.join(prediction_dir,'predictions.txt')

    # Check if file exists
    if not os.path.isfile(prediction_file):
        print('[-] Test prediction file not found!')
        print(prediction_file)
        sys.exit("Couldn't read predictions")
    # Get predictions
    pred_df = parse_prediction_file(prediction_file)
    
    # Get solutions
    file_list = os.listdir(solutions)
    print(f"files in {solutions}: {file_list}")
    # Check if file exists --- should generally be file_list[0], there's a '__MACOSX' showing up from zipping
    if "ref_val.csv" in file_list:
        solution_file = os.path.join(solutions,'ref_val.csv')
        sol_df_A, sol_df_mimic = parse_solution_file(solution_file)
        print("got solutions with 'ref_val.csv'")
    elif "ref_test.csv" in file_list:
        solution_file = os.path.join(solutions,'ref_test.csv')
        sol_df_A, sol_df_mimic = parse_solution_file(solution_file)
        print("got solutions with 'ref_test.csv'")
    elif len(file_list) > 0:
        solution_file = os.path.join(solutions, file_list[0])
        sol_df_A, sol_df_mimic = parse_solution_file(solution_file)
        print(f"got solutions with {file_list[0]}")
    else:
        sys.exit(f"Couldn't find solution file, have {len(file_list)} files in {solutions}")

    
    mimic_scores = get_scores(pred_df=pred_df, sol_df=sol_df_mimic, mm_vals = False)
    
    A_scores = get_scores(pred_df=pred_df, sol_df=sol_df_A, mm_vals = True)
    
    # Create ouput directory
    os.makedirs(pathlib.Path(output_dir).parent.resolve(), exist_ok=True)    
    output_file = os.path.join(output_dir, "scores.json")

    # Write scores
    save_scores(output_file, A_scores, mimic_scores)
