#!/bin/bash
: <<'END_COMMENT'
1. If run in docker:
docker pull icreateadockerid/anomaly_challenge
docker run -it -v [repo path]:/codabench icreateadockerid/anomaly_challenge:cpu /bin/bash
cd codabench

Otherwise, skip this step and directly go to step 2.

2. create running env with conda, venv, etc.
conda create --name [name] python=3.10
conda activate [name]

3. put the script under git repo, edit and run the script
chmod +x run.sh
./run.sh
END_COMMENT

## create folder structure
export data_split="test"
export task_folder="test_bioclip"
export ref_path="/home/wu.5686/imageo/challenge/reference_data/ref_$data_split.csv"

mkdir -p sample_result_submission/$task_folder/ref
mkdir -p sample_result_submission/$task_folder/res
cp $ref_path sample_result_submission/$task_folder/ref

: <<'END_COMMENT'
now, the folder structure for task is as below:
sample_result_submission
- task_folder
-- ref
--- ref.csv
-- res
END_COMMENT

## get the predictions
export input_dir="input_data/$data_split"
# export input_dir="/local/scratch/wu.5686/anomaly_challenge/input_data/dev_A"
export output_dir="sample_result_submission/$task_folder/res"
export program_dir="ingestion_program"
# export submission_dir="baselines/DINO_SGD_code_submission"
export submission_dir="baselines/BioCLIP_code_submission"
python ingestion_program/ingestion.py $input_dir $output_dir $program_dir $submission_dir

## score the predictions
export input_dir="sample_result_submission/$task_folder"
export output_dir="sample_result_submission/$task_folder"
python scoring_program/score_combined.py $input_dir $output_dir





