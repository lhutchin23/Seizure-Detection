import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import dataloader
import model
import CWTdenoisinganalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# Training using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

#using training data, can validate using preprocessed_data[2] and preprocessed_data[3], can test using preprocessed_data[4] and preprocessed_data[5]
preprocessed_data = dataloader.preprocess()
X_train_raw, y_train_raw = preprocessed_data[0], preprocessed_data[1]
X_test_raw, y_test_raw = preprocessed_data[4], preprocessed_data[5]


x_train = X_train_raw
y_train = torch.from_numpy(y_train_raw).long()
x_test = X_test_raw
y_test = torch.from_numpy(y_test_raw).long()


threshold_names = ['no denoise', 'sqwtolog', 'rigrsure', 'heuresure', 'visushrink']
results = {}

print('Data loading finished')

for name in threshold_names:
    print(f"Testing with thresholding method: {name}")
    
    if name == 'no denoise':
        t_func = 'no denoise'
    else:
        t_func = getattr(CWTdenoisinganalysis, name)

    print("Generating training scalograms")
    scalograms_np = CWTdenoisinganalysis.cwt_denoising(
        x_train, 
        wavelet="morl", 
        threshold_func=t_func, 
        threshold_type='scale_dependent', 
        mode='soft'
    )

    scalograms = torch.tensor(scalograms_np, dtype=torch.float32).unsqueeze(1)
    train_dataset = TensorDataset(scalograms, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print("Generating testing scalograms")
    scalograms_np_test = CWTdenoisinganalysis.cwt_denoising(
        x_test, 
        wavelet="morl", 
        threshold_func=t_func, 
        threshold_type='scale_dependent', 
        mode='soft'
    )

    scalograms_test = torch.tensor(scalograms_np_test, dtype=torch.float32).unsqueeze(1)
    test_dataset = TensorDataset(scalograms_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

  
    cnn = model.EEG_CNN().to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.00067)  #previsou runs on 0.001 learning rate showed signs of overshooting, swinging from an error of 0.2 to 0.01. 
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
            if batch_count % 50 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_count} | Loss: {loss.item():.4f}")

    cnn.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        eval_loader = test_loader
        for batch_X, batch_y in eval_loader:
            batch_X = batch_X.to(device)
            preds = cnn(batch_X).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
 #my computer training on CPU because it is a potato
            
    acc = accuracy_score(y_test.cpu().numpy(), all_preds)
    
    results[name] = acc
    print(f"{name:6} → {acc:.4%}")

    #generating confusion matrix for every threshold. 

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Seizure'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - {name} Denoising')
    save_path = os.path.join("results_12_09_25", f'confusion_matrix_{name}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    



