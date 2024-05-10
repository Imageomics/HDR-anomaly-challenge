# Butterfly Hybrid Detection: Evaluation
This is an anomaly detection problem, where the goal is to find hybrid butterflies. [Add more info]. 

There are 2 phases:
1. **Development phase:**
	* The provided training data contains images of all Species A subspecies with a signal set of the "Common hybrids" (the most common hybrids). 
	* The goal is to train a model to detect hybrid butterflies of Species A subspecies among their parent subspecies and mimics of the Species A Common hybrid among its two parent subspecies.
	* Upload your model; feedback will be provided on the validation performance for each component until the end of the challenge:
		1. Major and Minor (less common) hybrid subspecies of Species A. 
		2. Species B mimics of the Major hybrids and their parent subspecies.  
	* (Do we want to keep this condition as in the iris sample so it doesn't wind up dominated by one or two participants?:) Only the performance of a participant's MOST RECENT submission on the validation sets will be displayed on the leaderboard.
2. **Final phase:**
	* Automatic at the end of the challenge. Be sure to submit your preferred model as a final submission prior to the end of the challenge, as this will be the model run on the test data for final scores.
	* Each participant's latest submission will be evaluated on the final test sets and scores will be posted to the leaderboard. 

This competition allows you to submit a pre-trained prediction model which will be run on the validation and test datasets through CodaBench.
The submissions are evaluated based on the number of correctly identified hybrids assuming 95% accuracy on recognizing non-hybrids (improve this description to be more technically clear).
