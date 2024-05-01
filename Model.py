import os 
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from utils import get_params, get_vocab
import random as rnd

class NER(nn.Module):
  def __init__(self, embedding_dim=2304, hidden_size=50, n_classes=28):
    """
    The constructor of our NER model
    Inputs:
    - vacab_size: the number of unique words
    - embedding_dim: the embedding dimension
    - n_classes: the number of final classes (tags)
    """
    super(NER, self).__init__()
    ####################### TODO: Create the layers of your model #######################################

    # (2) Create an LSTM layer with hidden size = hidden_size and batch_first = True
    self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidierctional=True)

    # (3) Create a linear layer with number of neorons = n_classes
    self.linear = nn.Linear(hidden_size, n_classes)
    #####################################################################################################

  def forward(self, embedding):
    """
    This function does the forward pass of our model
    Inputs:
    - sentences: tensor of shape (batch_size, max_length)

    Returns:
    - final_output: tensor of shape (batch_size, max_length, n_classes)
    """

    final_output = None
    ######################### TODO: implement the forward pass ####################################
    
    hidden_output, _ = self.lstm(embedding)
    final_output = self.linear(hidden_output)
    ###############################################################################################
    return final_output, hidden_output
  
def train(model, train_dataset, batch_size=512, epochs=5, learning_rate=0.01):
    """
    This function implements the training logic
    Inputs:
    - model: the model ot be trained
    - train_dataset: the training set of type NERDataset
    - batch_size: integer represents the number of examples per step
    - epochs: integer represents the total number of epochs (full training pass)
    - learning_rate: the learning rate to be used by the optimizer
    """

    ############################## TODO: replace the Nones in the following code ##################################

    # (1) create the dataloader of the training set (make the shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # (2) make the criterion cross entropy loss
    criterion = torch.nn.CrossEntropyLoss()

    # (3) create the optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # GPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):

            # (4) move the train input to the device
            train_input = train_input.to(device)

            # (5) move the train label to the device
            train_label = train_label.to(device)


            # (6) do the forward pass
            output, _ = model(train_input)
            
            # (7) loss calculation (you need to think in this part how to calculate the loss correctly)
            # -1 is ignore 
            batch_loss = criterion(output.view(-1, len(tag_map)), train_label.view(-1)) # tegmap = vocab, check RNN


            # (8) append the batch loss to the total_loss_train
            total_loss_train += batch_loss.item()
            
            # (9) calculate the batch accuracy (just add the number of correct predictions)
            acc = (output.argmax(dim=-1) == train_label).sum().item()
            total_acc_train += acc

            # (10) zero your gradients
            optimizer.zero_grad()

            # (11) do the backward pass
            batch_loss.backward()

            # (12) update the weights with your optimizer
            optimizer.step()
            
            # epoch loss
            epoch_loss = total_loss_train / len(train_dataset)

            # (13) calculate the accuracy
            epoch_acc = total_acc_train / (len(train_dataset) * train_dataset[0][0].shape[0])

            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {epoch_loss} \
                | Train Accuracy: {epoch_acc}\n')

##############################################################################################################