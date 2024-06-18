#!/usr/bin/env python

# Scoring program for the HDR Anomaly Challenge

import os
import pathlib
import sys
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, accuracy_score

# Constants
HELPER_DIRECTORY = str(pathlib.Path(__file__).parent.resolve() / "helper_scripts")

# Importing functions from helper_scripts
sys.path.append(HELPER_DIRECTORY)
from dataio import parse_prediction_file, parse_solution_file, save_scores, parse_yaml_file, write_yaml_file

'''
# This is for local testing only, along with DEFAULT_CONFIG defined above
def get_config():
    config_path = DEFAULT_CONFIG
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        
    return parse_yaml_file(config_path)
'''

def evaluate_prediction(scores, labels, reversed=False):
    combined = list(zip(scores, labels))
    combined = sorted(combined, key=lambda x: x[0], reverse=reversed)
    combined = np.array(combined).astype(np.float32)
    threshold_i = None
    threshold_preds = None
    for i in range(combined.shape[0] + 1):
        ls = len(combined[:i])
        rs = len(combined[i:])
        preds = np.concatenate((np.zeros(ls), np.ones(rs)))
        recall = recall_score(combined[:, 1], preds, pos_label=0)
        if recall >= 0.95:
            threshold_i = i
            threshold_preds = preds
            break
    
    return threshold_preds, combined[:, 1]

def evaluate(scores, labels, reversed=False):
    """Requires lower score to mean more likely to be non-hybrid,
    and higher score to mean more likely to be hybrid.
    
    If you would like this to be reversed, set reversed=True
    """
    
    preds, gt = evaluate_prediction(scores, labels, reversed)
    h_recall = recall_score(gt, preds, pos_label=1)
    h_precision = precision_score(gt, preds, pos_label=1)
    f1 = f1_score(gt, preds, pos_label=1)
    roc_auc = roc_auc_score(gt, preds)
    acc = accuracy_score(gt, preds)

    return h_recall, h_precision, f1, roc_auc, acc

def evaluate_major_minor_prediction(pred_vals, labels, mm_vals, reversed=False):
    # TODO: the logic here sorts the preds and lables for evaluation
    # Then we sort again in order to align the mm_vals...
    # While likely correct here, there's probably a lot of redundancy
    # that could be cleaned up.
    print("Evaluating performance on signal vs non-signal hybrids")
    preds, gt = evaluate_prediction(pred_vals, labels, reversed=reversed)
    tmp = list(zip(pred_vals, labels, mm_vals))
    tmp = sorted(tmp, key=lambda x: x[0], reverse=reversed)
    sorted_mm_vals = np.array(tmp)[:, 2]
    # We are only looking at major and minor subspecies and nothing else, 
    # so there are only two options.
    major_idx = np.nonzero((sorted_mm_vals  == '1') & (gt == 1))
    minor_idx = np.nonzero((sorted_mm_vals == '0') & (gt == 1))
    maj_acc = accuracy_score(gt[major_idx], preds[major_idx])
    min_acc = accuracy_score(gt[minor_idx], preds[minor_idx])
    
    scores = {
        "major_recall" : maj_acc,
        "minor_recall" : min_acc
    }
    print(scores)
    
    return scores

def score_predictions(pred_vals, sol_gt_aligned, mm_vals_aligned=None, reverse_score_prediction=False):
    h_recall, h_precision, f1, roc_auc, acc = evaluate(pred_vals, sol_gt_aligned, reversed=reverse_score_prediction)
    
    scores = {
        "hybrid_recall" : float(h_recall),
        "hybrid_precision" : float(h_precision),
        "f1_score" : float(f1),
        "roc_auc" : float(roc_auc),
        "accuracy" : float(acc)
    }
    
    if mm_vals_aligned:
        mm_scores = evaluate_major_minor_prediction(pred_vals, sol_gt_aligned, mm_vals_aligned, reversed=reverse_score_prediction)
        scores.update(mm_scores)
    else:
        print("mm_vals not aligning")
        
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
    pred_filenames, pred_vals = parse_prediction_file(prediction_file)
    
    # Get solutions
    file_list = os.listdir(solutions)
    print(f"files in {solutions}: {file_list}")
    # Check if file exists --- should generally be file_list[0], there's a '__MACOSX' showing up from zipping
    if "ref_val_A.csv" in file_list:
        solution_file = os.path.join(solutions,'ref_val_A.csv')
        sol_filenames, sol_gt, mm_vals = parse_solution_file(solution_file)
        print("got solutions with 'ref_val_A.csv'")
    elif "ref_test_A.csv" in file_list:
        solution_file = os.path.join(solutions,'ref_test_A.csv')
        sol_filenames, sol_gt, mm_vals = parse_solution_file(solution_file)
        print("got solutions with 'ref_test_A.csv'")
    elif len(file_list) > 0:
        solution_file = os.path.join(solutions, file_list[0])
        sol_filenames, sol_gt, mm_vals = parse_solution_file(solution_file)
        print(f"got solutions with {file_list[0]}")
    else:
        sys.exit(f"Couldn't find solution file, have {len(file_list)} files in {solutions}")

    # Align predictions with solution via filename
    def align(ref_filenames, ref_vals):
        return [ref_vals[ref_filenames.index(filename)] for filename in pred_filenames]
    sol_gt_aligned = align(sol_filenames, sol_gt)
    mm_vals_aligned = align(sol_filenames, mm_vals)

    # Get scores
    scores = score_predictions(pred_vals, sol_gt_aligned, mm_vals_aligned) #, config.reverse_score_prediction)
    print(f"Full Scores: {scores}")
    
    # Create ouput directory
    os.makedirs(pathlib.Path(output_dir).parent.resolve(), exist_ok=True)    
    output_file = os.path.join(output_dir, "scores.json")

    # Write scores -- yaml probably unnecessary
    write_yaml_file(output_file, scores)
    save_scores(output_file, scores)