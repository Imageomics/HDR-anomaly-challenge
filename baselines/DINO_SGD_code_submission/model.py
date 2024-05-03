'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''

import numpy as np

class Model:
    def __init__(self):
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False

    def fit(self, X):
        # We may just require a dataloader, or a csv file
        pass
        

    def predict(self, X):
        # Again we may just send in a dataloader
        return np.zeros_like(X)