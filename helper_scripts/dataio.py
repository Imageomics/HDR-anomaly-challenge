import yaml

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
    camids, pred_vals = parse_delim_separated_text_file_as_columns(path)
    return camids, pred_vals

def parse_solution_file(path):
    camids, gt_vals = parse_delim_separated_text_file_as_columns(path)
    return camids, gt_vals

def parse_major_minor_file(path):
    camids, major_minor_vals = parse_delim_separated_text_file_as_columns(path)
    return camids, major_minor_vals