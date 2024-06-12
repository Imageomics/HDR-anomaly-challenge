# Butterfly Hybrid Detection: Evaluation
This challenge is an anomaly detection problem. The goal is to find hybrid butterfly instances.

Specifically, the test data comprise:
- A collection butterfly species' (denoted as Species A) subspecies. Participants can treat these subspecies as different object classes.
- Hybrids of these subspecies: these cases are produced by parents from different subspecies. Participants can treat hybrids with different combinations of parent subspecies as different object classes. We call the hybrid which is most common in the dataset the "signal" hybrid; it is the only hybrid provided for training.
- The goal of the challenge is to distinguish between non-hybrid instances and hybrid instances.

Moreover, Species A has a mimic (or co-mimic) Species B. These two butterfly species have quite similar appearances to evade predators. The test data thus further contains:
- A mimic butterfly species' (denoted as Species B) subspecies. Specifically, we only consider the two subspecies that correspond to the parent subspecies of Species A's signal hybrid.
- Hybrids of these two subspecies.
- The aim of this test set is to see whether an anomaly detection algorithm for butterflies is transferrable across mimic butterflies.

There are 2 phases:
1. **Development phase:**
	* The provided training data contains 1) images of all Species A's subspecies: these are considered normal (not anomaly) cases; 2) a signal set comprising the "signal" hybrid (anomaly) cases: the most common hybrids of Species A's subspecies in _this_ dataset.
	* The goal is to develop an algorithm to 1) detect hybrid butterflies of Species A's subspecies among their parent subspecies, and 2) detect the "signal" hybrid butterfly of a mimic Species (Species B) among its two parent subspecies.
	* Upload your model; feedback will be provided on the development set for each component until the end of the challenge:
		1. Signal and non-signal hybrid subspecies of Species A. 
		2. Signal hybrid subspecies of a mimic Species B.
	* Participants may submit _one_ score on the "development" sets to be displayed on the leaderboard. This score can be removed and replaced with a newer or better score as they choose.
2. **Final phase:**
	* Automatic at the end of the challenge. Be sure to submit your preferred algorithm as a final submission prior to the end of the challenge, as this will be the model run on the test data for final scores.
	* Each participant's added submission will be evaluated on the final test sets and scores will be posted to the leaderboard. 

Evaluation metric:

This competition allows you to submit your developed algorithm, which will be run on the development and test datasets through CodaBench.

Your algorithm needs to generate an anomaly score for each input image: the higher the score is, the more likely the input image is an anomaly (i.e., hybrid).

The submissions are evaluated based on two metrics:

1. The true positive rate (TPR) at the true negative rate (TNR) = 95%: the recall of hybrid cases, with a score threshold set to recognizing non-hybrid cases with 95% accuracy.
2. AUROC
