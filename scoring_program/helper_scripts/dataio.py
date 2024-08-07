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

    
# def parse_delim_separated_text_file_as_columns(path, delim=","):
#     with open(path, 'r') as f:
#         lines = f.readlines()
#         columns = zip(*[line.strip().split(delim) for line in lines])
#     return columns


def parse_prediction_file(path):
    return pd.read_csv(path, header=None, names=['filename','preds'])
    # filenames, pred_vals = parse_delim_separated_text_file_as_columns(path)
    # return pd.DataFrame({"filename": filenames, "preds": pred_vals})
    


def parse_solution_file(path):
    # hybrid stat is the 0-1 indicator
    df = pd.read_csv(path, dtype = {"hybrid_stat": np.int32})
    # Get mimic dataframe
    df_mimic = df.loc[df["ssp_indicator"] == "mimic"].copy()
    df_mimic["mm_vals"] = df_mimic["ssp_indicator"].map(lambda x: int(x == "major")).values.tolist()
    
    # Get Species A DataFrame and process it
    df_A = df.loc[df["ssp_indicator"] != "mimic"].copy()
    # This assumes that there are only 2 values ("major", "minor")
    df_A["mm_vals"] = df_A["ssp_indicator"].map(lambda x: int(x == "major")).values.tolist() 
    return df_A, df_mimic


def save_scores(path, A_scores, mimic_scores):
    score_record = {
        "A_score_major_recall": A_scores["major_recall"],
        "A_score_minor_recall": A_scores["minor_recall"],
        "A_PRC_AUC": A_scores["prc_auc"],
        "A_PRC_AUC_major": A_scores["major_prc_auc"],
        "A_PRC_AUC_minor": A_scores["minor_prc_auc"],
        "mimic_recall": mimic_scores["hybrid_recall"],
        "mimic_PRC_AUC": mimic_scores["prc_auc"]
    }
    print(f"Defined score record for leaderboard {score_record}")
    with open(path, "w") as f:
        f.write(json.dumps(score_record))
