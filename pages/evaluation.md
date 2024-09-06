# Butterfly Hybrid Detection: Evaluation
This challenge is an anomaly detection problem. The goal is to find hybrid butterfly instances. There are two sets of images considered in the test set, as described below.

## The First Component of the Test Dataset
This test data set comprises:
- An image collection of butterfly **Species A subspecies**. These are non-hybrid instances.
- An image collection of **Species A hybrids**, which are from parents from different subspecies of Species A. Note that in the training data, we only provide one type of hybrid (the "signal" hybrid), from one specific combination of parents. In the test data, there are also "non-signal" hybrids, which are from other combinations of parents. 

The goal of the challenge is to elicit new ways of solving this problem through designing a model to distinguish between non-hybrid instances and hybrid instances.

## The Second Component of the Test Dataset
Moreover, Species A has a [mimic](https://en.wikipedia.org/wiki/M%C3%BCllerian_mimicry) Species B. These two butterfly species have quite similar appearances to evade predators. Specifically, for the two subspecies of Species A that produce the signal hybrid, there are mimic subspecies of Species B which also hybridize. We thus provide a separate, second test data set that contains:
- An image collection of butterfly **Species B subspecies**. These are non-hybrid instances.
- An image collection of **Species B hybrids**, which are from parents from different subspecies of Species B. 

Specifically, we only consider one particular hybrid and two specific parent subspecies of Species B, corresponding to Species A signal hybrid and its two parent subspecies. 

This second test set aims to investigate whether the submitted anomaly detection algorithm for butterfly Species A is transferrable to the mimic butterfly Species B.

## Evaluation Phases
There are 2 phases. Each test data set is split into a development set and a final test set.  
1. **Development phase:**
	* The provided training data contains:
		- Images of all Species A subspecies: these images are considered "normal" (not anomaly) cases.
		- A signal set comprising the most common hybrid: these images are considered anomaly cases.*
	* The goal is to develop an algorithm to detect hybrid instances (the anomaly cases).
	* Upload your model: feedback will be provided on the development set until the end of the challenge; one submission is allowed per day.
		1. Detect signal and non-signal hybrid subspecies of Species A. 
		2. Detect subspecies hybrids among the mimic Species B (Species B subspecies are mimics of the Species A signal hybrid parent subspecies).
	* Participants may submit _one_ score on the development sets to be displayed on the leaderboard. This score can be removed and replaced with a newer or better score as they choose.
2. **Final phase:**
	* This phase will start automatically at the end of the challenge.
 	* Be sure to submit your preferred algorithm as a final submission before the end of the challenge, as this will be the model run on the test data for final scores.
	* Each participant's last submission will be evaluated on the final test set and scores will be posted to the leaderboard. 

## Evaluation metric

This competition allows you to submit your developed algorithm, which will be run on the development and the final test dataset through CodaBench.

Your algorithm needs to generate an anomaly score for each input image: the higher the score is, the more likely the input image is an anomaly (i.e., hybrid).

The submissions are evaluated based on two metrics:
- The true positive rate (TPR) at the true negative rate (TNR) = 95%: the recall of hybrid cases, with a score threshold set to recognizing non-hybrid cases with 95% accuracy.
- PRC AUC


\*  Note that these hybrids are just the most common within this particular dataset, not necessarily in general.
