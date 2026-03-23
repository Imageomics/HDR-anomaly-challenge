# Testing the Scoring Program
1. Create a conda environment for testing

```bash
conda create --name score-test python=3.10

conda activate score-test
```


2. Install all necessary dependencies

```bash
pip install pillow==10.3.0 tqdm==4.66.4 pandas==2.2.2 scikit-learn==1.4.2 ipykernel
```

3. Add environment to Jupyter Notebook

```bash
python -m ipykernel install --user --name score-test
```

4. Run [`validate.ipynb`](https://github.com/Imageomics/HDR-anomaly-challenge/blob/test_scoring/test_scoring_program/validate.ipynb) to test the scoring program. Note that validation results here are comparing expected scores to those produced with the provided input predictions; individual methods are not evaluated
