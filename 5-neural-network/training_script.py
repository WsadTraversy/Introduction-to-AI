from BCDataset import BCDataset
from torch.utils.data import DataLoader
from MLP import MLP
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy
import numpy as np

BATCH_SIZE = 64

train_datasets = BCDataset('combined')
validation_datasets = BCDataset('validation')
train_loader = DataLoader(dataset=train_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
validation_loader = DataLoader(dataset=validation_datasets, batch_size=validation_datasets.__len__(), shuffle=True, num_workers=2)

def calculate_gradient_norms(model=MLP(30, 8)):
    norms = list()
    for layer in model.hidden:
        norm = layer.weight.grad.norm().item()
        norms.append(norm)
    norm = model.output.weight.grad.norm().item()
    norms.append(norm)
    return norms

def model_train(model=MLP(h_size=1, neuron_number=4, dropout=False)):
    n_epochs = 100
    best_validation = np.inf
    best_training = np.inf
    best_val_acc = 0
    best_train_acc = 0
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4) # L2
    acc = BinaryAccuracy(multidim_average="global")
    for epoch in range(n_epochs):
        model.train() # Set the model to training mode
        total_training_loss = []
        total_training_acc = []
        # Training loop
        for _, (inputs, targets) in enumerate(train_loader):
            outputs = model(inputs) # Forward pass
            loss = loss_fn(outputs, targets) # Compute the loss
            total_training_loss.append(loss.item())
            optimizer.zero_grad() # Reset the gradients
            loss.backward() # Backward pass
            total_training_acc.append(acc(outputs, targets))
            optimizer.step() # Update the weights
        total_training_loss = np.mean(total_training_loss)
        total_training_acc = np.mean(total_training_acc)
        # Validation loop
        model.eval() # Set the model to evaluation mode
        total_valid_loss = []
        total_valid_acc = []
        for inputs, targets in validation_loader:
            with torch.no_grad():
                outputs = model(inputs) # Forward pass
                valid_loss = loss_fn(outputs, targets) # Compute the loss
                total_valid_loss.append(valid_loss.item())
                total_valid_acc.append(acc(outputs, targets))
        total_valid_loss = np.mean(total_valid_loss)
        total_valid_acc = np.mean(total_valid_acc)

        if total_valid_acc > best_val_acc or (total_training_loss <= best_training and total_valid_acc >= best_val_acc):
            best_val_acc = total_valid_acc
            best_train_acc = total_training_acc
            best_validation = total_valid_loss
            best_training = total_training_loss
            best_weights = model.state_dict()
        print(f"Epoch {epoch+1}/{n_epochs}, Training Loss: {total_training_loss:.4f}, Training Accuracy={total_training_acc:.4f}; Validation Loss: {total_valid_loss:.4f}, Validation Accuracy={total_valid_acc:.4f}")
    # Restore best model
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), "data/state_dict.pickle")
    # norms = calculate_gradient_norms(model)
    # average_gradient_norms = [np.mean(layer_norms) for layer_norms in norms]
    # for i, avg_norm in enumerate(average_gradient_norms):
    #     print(f'Layer {i}: Average Gradient Norm = {avg_norm}')
    print(f"Best training: {best_training:.4f}, Best training accuracy: {best_train_acc:.4f}, Best validation: {best_validation:.4f}, Best validation accuracy: {best_val_acc:.4f}")
    return model
