# Butterfly Hybrid Detection: Evaluation
This is an anomaly detection problem, where the goal is to find hybrid butterflies. [Add more info]. 

There are 2 phases:
1. **Development phase:**
	* The provided training data contains images of all Species A subspecies with a signal set, the "signal hybrids" (the most common hybrids in _this_ dataset). 
	* The goal is to train a model to detect hybrid butterflies of Species A subspecies among their parent subspecies and mimics of the Species A Signal hybrid among its two parent subspecies.
	* Upload your model; feedback will be provided on the validation performance for each component until the end of the challenge:
		1. Signal and non-signal hybrid subspecies of Species A. 
		2. Species B mimics of the Signal hybrids and their parent subspecies.  
	* Participants may submit _one_ score on validation sets to be displayed on the leaderboard. This score can be removed and replaced with a newer or better score as they choose.
2. **Final phase:**
	* Automatic at the end of the challenge. Be sure to submit your preferred model as a final submission prior to the end of the challenge, as this will be the model run on the test data for final scores.
	* Each participant's added submission will be evaluated on the final test sets and scores will be posted to the leaderboard. 

This competition allows you to submit a pre-trained prediction model which will be run on the validation and test datasets through CodaBench.
The submissions are evaluated based on the number of correctly identified hybrids assuming 95% accuracy on recognizing non-hybrids (improve this description to be more technically clear).
