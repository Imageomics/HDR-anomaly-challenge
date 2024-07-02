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
    filenames, pred_vals = parse_delim_separated_text_file_as_columns(path, delim=",")
    return pd.DataFrame({"filename": filenames, "preds": pred_vals})


def parse_solution_file(path):
    # hybrid stat is the 0-1 indicator
    df = pd.read_csv(path, dtype = {"hybrid_stat": np.int32})
    # Get mimic dataframe
    df_mimic = df.loc[df["ssp_indicator"] == "mimic"].copy()
    
    # Get Species A DataFrame and process it
    df_A = df.loc[df["ssp_indicator"] != "mimic"].copy()
    # This assumes that there are only 2 values ("major", "minor")
    df_A["mm_vals"] = df_A["ssp_indicator"].map(lambda x: int(x == "major")).values.tolist() 
    return df_A, df_mimic


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
