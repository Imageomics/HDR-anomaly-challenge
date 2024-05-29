#!/usr/bin/env python

# Scoring program for the HDR Anomaly Challenge

import os
import pathlib
import sys

import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, accuracy_score

# Constants
DEFAULT_CONFIG = pathlib.Path(__file__).parent.resolve() / "scoring_config.yaml"
HELPER_DIRECTORY = str(pathlib.Path(__file__).parent.parent.resolve() / "helper_scripts")

# Importing functions from helper_scripts
sys.path.append(HELPER_DIRECTORY)
from dataio import parse_yaml_file, parse_prediction_file, parse_solution_file, parse_major_minor_file, write_yaml_file

def get_config():
    config_path = DEFAULT_CONFIG
    if len(sys.argv) > 1:
        config_path = argv[1]
        
    return parse_yaml_file(config_path)

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

def evaluate_major_minor_prediction(scores, labels, camids, major_cams, minor_cams, reversed=False):
    
    preds, gt = evaluate_prediction(scores, labels, reversed=reversed)
    tmp = list(zip(scores, labels, camids))
    tmp = sorted(tmp, key=lambda x: x[0], reverse=reversed)
    sorted_camids = np.array(tmp)[:, 2]
    major_idx = np.isin(np.array(sorted_camids), major_cams)
    minor_idx = np.isin(np.array(sorted_camids), minor_cams)
    maj_acc = accuracy_score(gt[major_idx], preds[major_idx])
    min_acc = accuracy_score(gt[minor_idx], preds[minor_idx])
    
    return maj_acc, min_acc

def score_predictions(pred_camids, pred_vals, sol_gt_aligned, reverse_score_prediction=False, mm_vals_aligned=None):
    h_recall, h_precision, f1, roc_auc, acc = evaluate(pred_vals, sol_gt_aligned, reversed=reverse_score_prediction)
    
    scores = {
        "hybrid_recall" : float(h_recall),
        "hybrid_precision" : float(h_precision),
        "f1_score" : float(f1),
        "roc_auc" : float(roc_auc),
        "accuracy" : float(acc)
    }
    
    if mm_vals_aligned:
        pass
        # TODO:
        # Call evaluate_major_minor_prediction and add to scores
        # NOTE: will have to handle incomplete list (for example: the mm_vals_aligned will not match the shape of all pred_vals)
        # it's a subset
        #evaluate_major_minor_prediction()
        
    return scores

if __name__ == "__main__":

    # Get scoring configurations
    config = get_config()
    
    # Get predictions
    pred_camids, pred_vals = parse_prediction_file(config.input_data)
    
    # Get solutions
    sol_camids, sol_gt = parse_solution_file(config.solution_data)
    
    # Align predictions with solution via CAMID
    def align(ref_camids, ref_vals):
        return [ref_vals[ref_camids.index(camid)] for camid in pred_camids]
    sol_camids, sol_gt
    sol_gt_aligned = align(sol_camids, sol_gt)
    
    # Optionally get major-minor list
    mm_vals_aligned = None
    if hasattr(config, 'major_minor_data') and config.major_minor_data:
        report_major_minor_score = True
        mm_camids, mm_vals = parse_major_minor_file(config.major_minor_data)
        mm_vals_aligned = align(mm_camids, mm_vals)
        
    # Get scores
    scores = score_predictions(pred_camids, pred_vals, sol_gt_aligned, config.reverse_score_prediction, mm_vals_aligned)
    print(scores)
    
    # Create ouput directory
    os.makedirs(pathlib.Path(config.output_data).parent.resolve(), exist_ok=True)
    
    # Write scores
    write_yaml_file(config.output_data, scores)

