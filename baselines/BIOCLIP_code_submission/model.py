'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
from open_clip import create_model
from tqdm import tqdm
import torch
import numpy as np

class Model:
    def __init__(self, device, batch_size=8, lp_config='sgd'):
        self.batch_size=batch_size
        self.device=device
        model = create_model("hf-hub:imageomics/bioclip", output_dict=True, require_pretrained=True)
        self.model = model.to(device)
        self.lp_config=lp_config

    def get_feats_and_meta(self, dloader, ignore_feats=False):
        all_feats = None
        labels = []
        camids = []
        for img, lbl, meta, _ in tqdm(dloader, desc="Extracting features"):
            with torch.no_grad():
                feats = None
                if not ignore_feats:
                    out = self.model(img.to(self.device))['image_features']
                    feats = out.detach().cpu().numpy()
                
            if all_feats is None:
                all_feats = feats
            else:
                all_feats = np.concatenate((all_feats, feats), axis=0)
                
            labels.extend(lbl.detach().cpu().numpy().tolist())
            camids.extend(list(meta))
            
        labels = np.array(labels)
                
        return all_feats, labels, camids

    def linear_probing(self, X, y, classifier_config="sgd"):
        from sklearn.svm import SVC
        from sklearn.linear_model import SGDClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        non_hybrid_weight = 1
        hybrid_weight = 1

        class_weights = {0: non_hybrid_weight, 1: hybrid_weight}

        if classifier_config == "svm":
            clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1, class_weight=class_weights, probability=True))
        elif classifier_config == "sgd":
            clf = SGDClassifier(loss="log_loss", alpha=0.001, penalty="l2", eta0=0.001, n_iter_no_change=100,
                                learning_rate='constant', max_iter=10000, class_weight=class_weights)
        elif classifier_config == "knn":
            clf = KNeighborsClassifier(n_neighbors=2)
        elif classifier_config == "gaussian":
            clf = GaussianProcessClassifier(random_state=0)
            
        clf.fit(X, y)

        preds = clf.predict(X)
        correct = preds == y

        hybrid_correct = correct[y == 1].sum()
        non_hybrid_correct = correct[y == 0].sum()
        #print(hybrid_correct)
        #print(non_hybrid_correct)

        train_acc = clf.score(X, y)

        train_h_acc = hybrid_correct / (y==1).sum()
        train_nh_acc = non_hybrid_correct / (y==0).sum()
        
        return clf, train_acc, train_h_acc, train_nh_acc

    
    def fit(self, train_dl):
        # We may just require a dataloader, or a csv file
        train_features, train_labels, _ = self.get_feats_and_meta(train_dl)
        self.clf, _,_,_ = self.linear_probing(train_features, train_labels, self.lp_config)
        

    def predict(self, test_dl):
        # Again we may just send in a dataloader
        test_features, test_labels, test_camids = self.get_feats_and_meta(test_dl)

        scores = self.clf.predict_proba(test_features)[:, 1]

        return scores