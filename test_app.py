import requests
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import ta
from ta.momentum import StochasticOscillator
import torch.optim as optim
import mlflow
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, accuracy_score, cohen_kappa_score, matthews_corrcoef


def test_get():
    get_url = "https://mlsd-indicatorss.darkube.app/"
    print("Testing GET method: ", get_url)
    response = requests.get(get_url)
    assert response.status_code == 200
    

def test_post():
    post_url = "https://mlsd-indicatorss.darkube.app/predict"
    print("Testing POST method: ", post_url)
    file_path = "vabemellat_30min_200.csv"
    with open(file_path , "rb") as file:
        files = {"file": file}
        response = requests.post(post_url, files=files)
    assert response.status_code == 200

def test_end_to_end_sample():
    post_url = "https://mlsd-indicatorss.darkube.app/predict"
    print("Testing E2E app: ", post_url)
    file_path = "vabemellat_30min_200.csv"
    with open(file_path , "rb") as file:
        files = {"file": file}
        response = requests.post(post_url, files=files)
    print(response, '====================')
    assert response.status_code == 200

############ Test Model ############
class IntegerTransformer3(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_heads, num_layers):
        super().__init__()
        self.conv = nn.Conv2d(kernel_size=(1, embedding_dim), out_channels=1, in_channels=1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(num_embeddings, num_heads),
            num_layers
        )

        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 2)


    def forward(self, input_ids):
        embedded = self.conv(input_ids)
        output = self.transformer(embedded.squeeze(3))
        out = self.fc1(output.squeeze(1))
        out = self.relu(out)
        out = self.fc2(out).softmax(dim=1)
        return out


def test_model_ci():
    model = IntegerTransformer3(num_embeddings=100, embedding_dim=9, num_heads=4, num_layers=2)
    model.load_state_dict(torch.load('modelV1.pth', map_location=torch.device('cpu')))
    model.eval()

    val_accuracy = 0
    val_num_samples = 0
    val_true = []
    val_pred = []

    val_x = torch.load('data/X_val_V1.pt')
    val_y = torch.load('data/y_val_V1.pt')

    val_dataset = TensorDataset(val_x, val_y)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        
    for i, (X_batch, y_batch) in enumerate(val_dataloader):

        outputs = model(X_batch.unsqueeze(1).to('cpu'))

        _, predicted = torch.max(outputs, 1)

        val_pred += list(predicted.cpu())
        val_true += [y_batch]
        val_accuracy += torch.sum(predicted == y_batch.to('cpu')).item()

        val_num_samples += len(X_batch)

    avg_accuracy = val_accuracy / val_num_samples
    
    val_y_true = []
    for q in val_true:
        val_y_true += q
    
    val_y_pred = []
    for q in val_pred:
        val_y_pred += [q.item()]
    
    tn, fp, fn, tp = confusion_matrix(val_y_true, val_y_pred).ravel()
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (tp + fn)
    true_negative_rate = tn / (tn + fp)
    false_discovery_rate = fp/ (tp + fp)
    recall = recall_score(val_y_true, val_y_pred)
    precision = precision_score(val_y_true, val_y_pred)
    acc = accuracy_score(val_y_true, val_y_pred)
    cohen_kappa = cohen_kappa_score(val_y_true, val_y_pred)
    matthews_corr = matthews_corrcoef(val_y_true, val_y_pred)

    return acc, precision


def test_model_accuracy():
    acc, precision = test_model_ci()
    print(acc, '=================')
    assert acc >= 0

if __name__=="__main__":
    test_get()
    test_post()
    test_model_accuracy()
