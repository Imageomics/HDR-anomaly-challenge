import json
import numpy as np
import pandas as pd

class DictToObj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def __str__(self):
        out_str = "=====Object Values=====\n"
        for k, v in self.__dict__.items():
            out_str += f"{k}: {str(v)}\n"
        out_str = out_str[:-1]
        return out_str

    
def parse_delim_separated_text_file_as_columns(path, delim=" "):
    with open(path, 'r') as f:
        lines = f.readlines()
        columns = zip(*[line.strip().split(delim) for line in lines])
    return columns


def parse_prediction_file(path):
    filenames, pred_vals = parse_delim_separated_text_file_as_columns(path)
    return filenames, pred_vals, pd.DataFrame({"filename": filenames, "preds": pred_vals})


def parse_solution_file_A(path):
    df = pd.read_csv(path, dtype = {"hybrid_stat": np.int32})
    df = df.loc[df["ssp_indicator"] != "mimic"].copy()
    filenames = df["filename"].values.tolist()
    gt_vals = list(df["hybrid_stat"])
    # This assumes that there are only 2 values ("major", "minor")
    is_major_vals = df["ssp_indicator"].map(lambda x: int(x == "major")).values.tolist() 
    return filenames, gt_vals, is_major_vals, df


def parse_solution_file_mimic(path):
    df = pd.read_csv(path, dtype = {"hybrid_stat": np.int32})
    df = df.loc[df["ssp_indicator"] == "mimic"].copy()
    filenames = df["filename"].values.tolist()
    gt_vals = list(df["hybrid_stat"])
    return filenames, gt_vals


def parse_major_minor_file(path):
    filenames, major_minor_vals = parse_delim_separated_text_file_as_columns(path)
    return filenames, major_minor_vals


def save_scores(path, A_scores, mimic_scores):
    score_record = {
        "A_score_major": A_scores["major_recall"],
        "A_score_minor": A_scores["minor_recall"],
        "A_AUC": A_scores["roc_auc"],
        "mimic_score": mimic_scores["hybrid_recall"],
        "mimic_AUC": mimic_scores["roc_auc"]
    }
    print(f"Defined score record for leaderboard {score_record}")
    with open(path, "w") as f:
        f.write(json.dumps(score_record))
