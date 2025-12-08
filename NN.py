import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import dataloader
import model
import CWTdenoisinganalysis
import time

# Training using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

#using training data, can validate using preprocessed_data[2] and preprocessed_data[3], can test using preprocessed_data[4] and preprocessed_data[5]
preprocessed_data = dataloader.preprocess()
X_train_raw, y_raw = preprocessed_data[2], preprocessed_data[3]

if isinstance(y_raw, np.ndarray):
    y_train = torch.from_numpy(y_raw).long()
else:
    y_train = y_raw.long()

if isinstance(X_train_raw, torch.Tensor):
    X_train = X_train_raw.numpy()
else:
    X_train = X_train_raw

threshold_names = ['no denoise', 'sqwtolog', 'rigrsure', 'heuresure']
results = {}

print('Data loading finished')

for name in threshold_names:
    print(f"Testing with thresholding method: {name}")
    
    if name == 'no denoise':
        t_func = 'no denoise'
    else:
        t_func = getattr(CWTdenoisinganalysis, name)

    print("Generating scalograms")
    scalograms_np = CWTdenoisinganalysis.cwt_denoising(
        X_train, 
        wavelet="morl", 
        threshold_func=t_func, 
        threshold_type='scale_dependent', 
        mode='soft'
    )

    scalograms = torch.tensor(scalograms_np, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(scalograms, y_train)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

  
    cnn = model.EEG_CNN().to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training")

    #15 epochs, usually for data like this 10-20 epochs should be enough
    for epoch in range(15):
        cnn.train()
        epoch_loss = 0
        batch_count = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = cnn(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_count} | Loss: {loss.item():.4f}")

    cnn.eval()
    all_preds = []
    with torch.no_grad():
        eval_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        for batch_X, _ in eval_loader:
            batch_X = batch_X.to(device)
            preds = cnn(batch_X).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            
    acc = accuracy_score(y_train.cpu().numpy(), all_preds)
    
    results[name] = acc
    print(f"{name:6} → {acc:.4%}")

