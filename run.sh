#!/bin/bash
: <<'END_COMMENT'
1. 
1.a. If run in docker:

docker pull [image_id]

1.a.1. If use a GPU:
docker run -it --gpus device=0 -v [repo_path]:/codabench [image_id] /bin/bash
1.a.2. If only use CPU:
docker run -it -v [repo_path]:/codabench [image_id] /bin/bash

cd codabench

1.b. If run with conda env:

1.b.1. create running env with conda, venv, etc.
conda create --name [name] python=3.10
conda activate [name]
1.b.2. install all necessary dependencies
pip install pillow==10.3.0 tqdm==4.66.4 pandas==2.2.2 scikit-learn==1.4.2

2. edit and run the script

chmod +x run.sh

./run.sh
END_COMMENT


export task_type="folder; predict; evaluate" #folder; predict; evaluate
export data_split="dev"
export baseline_model="bioclip"
export task_folder="${data_split}_${baseline_model}"


# Create folder structure
if [[ "$task_type" == *"folder"* ]]; then
    mkdir -p sample_result_submission/$task_folder/ref
    mkdir -p sample_result_submission/$task_folder/res
    export ref_path="[the path of folder you put the simulated ground truth csv file]/ref_$data_split.csv"
    cp $ref_path sample_result_submission/$task_folder/ref # Put the simulated ground truth csv file in the ref folder.
fi

: <<'END_COMMENT'
now, the folder structure for task is as below:
sample_result_submission
- task_folder
-- ref
--- ref.csv
-- res
END_COMMENT

# Get the predictions
if [[ "$task_type" == *"predict"* ]]; then
    export input_dir="input_data/$data_split" # This is the directory you put the images in.
    export output_dir="sample_result_submission/$task_folder/res" # The prediction file will output to this directory.
    export program_dir="ingestion_program"
    if [ "$baseline_model" == "bioclip" ]; then
        export submission_dir="baselines/BioCLIP_code_submission"
    elif [ "$baseline_model" == "dino" ]; then
        export submission_dir="baselines/DINO_SGD_code_submission"
    else
        echo "$baseline_model is undefined"
        exit 1
    fi
    
    python3 ingestion_program/ingestion.py $input_dir $output_dir $program_dir $submission_dir
fi

# Score the predictions
if [[ "$task_type" == *"evaluate"* ]]; then
    export input_dir="sample_result_submission/$task_folder"
    export output_dir="sample_result_submission/$task_folder"
    python3 scoring_program/score_combined.py $input_dir $output_dir
fi