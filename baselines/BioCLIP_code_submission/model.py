'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
from open_clip import create_model
from torchvision import transforms
from tqdm import tqdm
import torch
import numpy as np
import pickle

class Model:
    def __init__(self, device='cuda'):
        self.device=device

        model = create_model("hf-hub:imageomics/bioclip", output_dict=True, require_pretrained=True)
        self.model = model.to(device)
                
        with open('/local/scratch/wu.5686/anomaly_challenge/model.pkl', 'rb') as f:
            self.clf = pickle.load(f)

        self.preprocess_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            ]
        )
    

    def predict(self, datapoint):
        with torch.no_grad():
            image = self.preprocess_img(datapoint).to(self.device)
            image_feature = self.model.encode_image(image.unsqueeze(0))
            image_feature = image_feature.detach().cpu().numpy()

            score = self.clf.predict_proba(image_feature)[:, 1][0]
        
        return score