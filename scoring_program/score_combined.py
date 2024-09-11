#!/usr/bin/env python

# Scoring program for the HDR Anomaly Challenge

import os
import pathlib
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from statistics import harmonic_mean
from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score, roc_auc_score, accuracy_score

# Constants

# Importing functions from helper_scripts

def parse_solution_file(path):
    # hybrid stat is the 0-1 indicator
    df = pd.read_csv(path, dtype = {"hybrid_stat": np.int32})
    # Get mimic dataframe
    df_mimic = df.loc[df["ssp_indicator"] == "mimic"].copy()    
    # Get Species A DataFrame
    df_A = df.loc[df["ssp_indicator"] != "mimic"].copy()

    return df_A, df_mimic

def evaluate_prediction(score_df):
    # loop predictions from most likely non-hybrid to most likeyly hybrid
    for threshold_pred in sorted(set(score_df["preds"])):
        score_df['converted_preds'] = score_df["preds"].apply(lambda x: 1 if x > threshold_pred else 0)
        threshold_recall = recall_score(score_df["hybrid_stat"], score_df["converted_preds"], pos_label=0)  # non-hybrid is the positive here, so positive label is 0
        if threshold_recall >= 0.95:
            break
    
    print(f'With non-hybrid recall {str(round(threshold_recall, 4))}, the predictions equal and lower than the threshold confident score {str(threshold_pred)} are all non-hybrids and the ones higher are all hyrids.')

    return score_df, threshold_recall, threshold_pred


def evaluate_major_minor_prediction(score_df):
    print("Evaluating performance on signal vs non-signal hybrids")
    # Set to compare just hybrids and look if they're predicted as such
    major_df = score_df.loc[score_df["ssp_indicator"] == "major"].copy()
    minor_df = score_df.loc[score_df["ssp_indicator"] == "minor"].copy()

    major_recall = recall_score(major_df["hybrid_stat"], major_df["converted_preds"])
    minor_recall = recall_score(minor_df["hybrid_stat"], minor_df["converted_preds"])

    major_roc_auc = roc_auc_score(major_df["hybrid_stat"], major_df["preds"])
    minor_roc_auc = roc_auc_score(minor_df["hybrid_stat"], minor_df["preds"])

    major_prc_auc = average_precision_score(major_df["hybrid_stat"], major_df["preds"])
    minor_prc_auc = average_precision_score(minor_df["hybrid_stat"], minor_df["preds"])
    
    scores = {
        "major_recall" : major_recall,
        "minor_recall" : minor_recall,
        "major_prc_auc" : major_prc_auc,
        "minor_prc_auc" : minor_prc_auc,
        "major_roc_auc" : major_roc_auc,
        "minor_roc_auc" : minor_roc_auc,
    }
    
    return scores

def score_predictions(score_df, mm_vals=False):

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
        scores.update(mm_scores)
        print(f"Full Scores Species A hybrid detection: {scores}")
    else:
        print(f"Full Scores Mimic hybrid detection: {scores}")
        
    return scores


def get_scores(pred_df=None, sol_df=None, mm_vals = False):
    # merge ref with predictions
    # aligns the ref values with scores in the columns based on filenames
    score_df = pd.merge(sol_df, pred_df, on = "filename", how = "inner")
    
    # Check aligned on all expected files
    if score_df.shape[0] != sol_df.shape[0]:
        sys.exit(f"There should have been {sol_df.shape[0]} predictions, but we only got {score_df.shape[0]}")
    
    scores = score_predictions(score_df, mm_vals) #, config.reverse_score_prediction)
    
    return scores

def save_scores(path, A_scores, mimic_scores):
    score_record = {
        "A_score_major_recall": A_scores["major_recall"],
        "A_score_minor_recall": A_scores["minor_recall"],
        "A_PRC_AUC": A_scores["prc_auc"],
        "A_PRC_AUC_major": A_scores["major_prc_auc"],
        "A_PRC_AUC_minor": A_scores["minor_prc_auc"],
        "mimic_recall": mimic_scores["hybrid_recall"],
        "mimic_PRC_AUC": mimic_scores["prc_auc"],
        "challenge_score": harmonic_mean([A_scores["major_recall"], A_scores["minor_recall"], mimic_scores["hybrid_recall"]])
    }
    print(f"Defined score record for leaderboard {score_record}")
    with open(path, "w") as f:
        f.write(json.dumps(score_record))

if __name__ == "__main__":

    print("We're running scoring")

    # Get the current UTC time
    current_time_utc = datetime.now(timezone.utc)
    # Print the timestamp in UTC
    print("Current UTC Time:", current_time_utc.strftime('%Y-%m-%d %H:%M:%S'))
    
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
    pred_df = pd.read_csv(prediction_file, header=None, names=['filename','preds'])
    
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
