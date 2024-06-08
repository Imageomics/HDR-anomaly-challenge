import yaml
import json
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
    
def write_yaml_file(path, data):
    with open(path, 'w') as f:
        yaml.dump(data, f)

def parse_yaml_file(path):
    with open(path, 'r') as f:
        yaml_dict = yaml.safe_load(f)
        return DictToObj(**yaml_dict)
    
def parse_delim_separated_text_file_as_columns(path, delim=" "):
    with open(path, 'r') as f:
        lines = f.readlines()
        columns = zip(*[line.strip().split(delim) for line in lines])
    return columns

def parse_prediction_file(path):
    filenames, pred_vals = parse_delim_separated_text_file_as_columns(path)
    return filenames, pred_vals

def parse_solution_file(path):
    df = pd.read_csv(path)
    filenames = df["filename"].values.tolist()
    gt_vals = df["hybrid_stat_ref"].map(lambda x: int(x == "hybrid")).values.tolist()
    return filenames, gt_vals

def parse_major_minor_file(path):
    filenames, major_minor_vals = parse_delim_separated_text_file_as_columns(path)
    return filenames, major_minor_vals


def save_scores(path, scores):
    ## TODO: This needs proper labels after re-write
    # Will return scores now, though they are NOT properly labeled
    score_record = {
        "A_score_major": scores["hybrid_recall"],
        "A_score_minor": scores["accuracy"],
        "A_AUC": scores["roc_auc"]
    }
    with open(path, "w") as f:
        f.write(json.dumps(score_record))
