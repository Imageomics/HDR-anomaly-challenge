# Starting Kit and Example Submission

### Starting Kit

Please see the [sample challenge repository](https://github.com/Imageomics/HDR-anomaly-challenge-sample) for example training code (related to the provided [baselines](https://github.com/Imageomics/HDR-anomaly-challenge/tree/main/baselines)) and general structure expected when your submission is released as open-source at the end of the competition.

### Example Submission Format

Your submission to the challenge should include a `model.py` with `model` class, the model weights, and a `requirements.txt`. If you have requirements not included in the [whitelist](https://github.com/Imageomics/HDR-anomaly-challenge/blob/main/ingestion_program/whitelist.txt), you may open an [issue](https://github.com/Imageomics/HDR-anomaly-challenge/issues) to request it (please check if someone else has requested your required package before making your own issue).
See the provided [baselines](https://github.com/Imageomics/HDR-anomaly-challenge/tree/main/baselines) for examples.

### Common Error

[!!] Do not zip the whole folder. ONLY select the `model.py` and relevant weight and requirements files to make the tarball.

<img src="https://github.com/user-attachments/assets/10b49a84-d42a-42c2-8855-e4b563b28b15" alt="common_error: no module named model" width="750">

The above error is mostly likely caused by zipping the whole folder (instead of zipping just the contents) when making the tarball.
