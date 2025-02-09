# Harnessing the Data Revolution (HDR) Anomaly ML Challenge
Repository for [Imageomics' CodaBench challenge](https://www.codabench.org/competitions/3764/) as part of the broader [HDR Anomaly ML Challenge](https://www.nsfhdr.org/mlchallenge).

Our challenge is on the problem of hybrid detection among two mimetic species of butterflies. The [training data](files/butterfly_anomaly_train.csv) provides images from one species (`Species A`), with a collection of `Signal hybrids` (hybrids of the two subspecies that have the greatest presence in our data--this larger presence is due to collection bias). There are then two subtasks on which the models submitted by participants are evaluated. In both instances, the model must detect the hybrids, but there is a more fine-grained analysis of the results:
1. For Species A images, the models are graded individually on their identification of Signal and non-Signal hybrids. All non-Signal hybrid parents are also seen in the training set, but the non-Signal hybrids themselves are unseen.
2. For the mimetic species, the models are graded on their ability to detect hybrids of the mimics of the two subspecies that birth the Signal hybrids.


We use the same base container as the challenges from the three other HDR Institutes; it is available on the [A3D3 Anomaly Challenge GitHub](https://github.com/a3d3-institute/HDRchallenge/pkgs/container/hdr-image). There are options to use packages not included in this container as long as they are in the [pre-approved list](ingestion_program/whitelist.txt); if your solution requires a package not listed there, please open an [issue](https://github.com/Imageomics/HDR-anomaly-challenge/issues) to request it (provided it is not already requested by someone else).

For more details on participating in the challenge, please see the pages in the [CodaBench challenge](https://www.codabench.org/competitions/3764/) itself.

## Full Bundle Structure

```
baselines/
    BioCLIP_code_submission/
        # Zip the contents of this folder to submit this baseline to the challenge on CodaBench
        clf.pkl
        metadata
        model.py
        requirements.txt
    DINO_SGD_code_submission/
        # Zip the contents of this folder to submit this baseline to the challenge on CodaBench
        clf.pkl
        metadata
        model.py
        requirements.txt
competition.yaml
files/
    butterfly_anomaly_train.csv
Imageomics_logo_butterfly.png
ingestion_program/
    ingestion.py
    metadata.yaml
    whitelist.txt
input_data/
    # Images used for testing submitted models (images must be zipped together directly under input_data, not within a subfolder)
    `val.zip`
    `test.zip`
pages/
    # These will be all the optional tabs under "Get Started" on the challenge page. They are used to describe the challenge for participants
    data.md
    evaluation.md
    overview.md
    starting_kit.md
    terms.md
reference_data/
    # Labels for the images in input_data/. Can be .txt, .csv, etc.
    `ref_val.csv`
    `ref_test.csv`
scoring_program/
    metadata.yaml
    score_combined.py
```

### Notes on Structure

- This is a challenge with two different scored objectives combined into one program (or `task`) to unify the leaderboard. If you're adapting this for your own challenge, the only thing to potentially change in `metadata.yaml` (under the scoring & ingestion programs) is the filename in `/app/program/<ingestion/score-file>.py`. The input and output to these files is handled by the CodaBench backend based on the information provided in `competition.yaml` for the tasks and phases. 
  - If you choose to have two tasks, it will run through each task once fully (from ingestion through scoring) before moving on to the next task--they are not connected to each other. This creates two separate leaderboards, hence the combination.
- The base container specified in the `competition.yaml` must have the requirements for the ingestion and scoring programs. These can be manually imported through a `requirements.txt`, but then each of these files will need to do so at the beginning. It is better to install these requirements in the base container and allow for participants to include a requirements file with their model's needed programs.
  - Example: For this challenge, our container must have (at minimum) `pillow`, `pandas`, and `tqdm`. The base container used for this competition will be provided.
  - Any requirements used by participants must be on the approved whitelist (or participants must reach out to request their addition) for security purposes.
- Scores must be saved to a `score.json` file where the keys detailed in the `Leaderboard` section of the `competition.yaml` are give as the keys for the scores.
- This full collection of files and folders is zipped as-is to upload the bundle to CodaBench.
- `run.sh` is a bash script to simulate the scoring process used for the Leaderboard on your local machine. This works by first building the docker container, and then running the bash script within the virtual environment. The script will
  -  Create a folder `/ref` for the csv ground truth file and a folder `/res` for the generated prediction txt file.
  -  Run `ingestion_program/ingestion.py` to get the predictions from your model and output the predictions to the txt file.
  -  Run `scoring_program/score_combined.py` to evaluate the predictions by comparing to the grouth truth. The final scores are then written to a json file.
- To test with `run.sh`, you should provide your own curated validation dataset (e.g. subsampling from train split) including images and a simulated ground truth file.
