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

4. Run the [testing notebook](https://github.com/Imageomics/HDR-anomaly-challenge/blob/test_scoring/test_scoring_program/validate.ipynb)
