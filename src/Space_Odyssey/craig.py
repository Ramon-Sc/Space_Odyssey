import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import torchvision.models as models
from data_preprocesssor import get_torch_datasets,get_config
import numpy as np
import pickle
from torch.utils.data import Subset
import csv

class craig():
    """
    Craig - coreset selection algorithm implementation
    Args:
        model: the model to train
        optimizer: the optimizer to use
        train_dataset: the training dataset of type torchvision.datasets.ImageFolder
        val_loader: the validation data loader of type torchvision.datasets.ImageFolder
        num_epochs_warmup: the number of epochs to train the model on the full dataset before starting the coreset selection
        num_epochs: the number of epochs to train for each coreset selection
        n_samples: the number of samples to select for each coreset selection
        n_reselections: the number of coreset selections to make during training
    """

    def __init__(self,model,optimizer,train_dataset,val_dataset,num_epochs_warmup,num_epochs,n_samples,n_reselections):

        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = 32
        #number of epochs per coreset selection
        self.num_epochs = num_epochs
        #number of samples per coreset selection
        self.n_samples =  n_samples
        #number of coreset selections
        self.n_reselections = n_reselections
        #number of epochs to train the model on the full dataset before starting the coreset selection
        self.num_epochs_warmup = num_epochs_warmup

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_model_warmup(self):
        
        #train the model on the full dataset for num_epochs_warmup epochs before starting the coreset selection

        #create a data loader for the full dataset
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        self.model.train()

        #move the model to gpu
        self.model.to(self.device)

        for epoch in range(self.num_epochs_warmup):
            print(f"Epoch {epoch+1}/{self.num_epochs_warmup}")
            for indices, inputs, labels in train_loader:

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                loss.backward()
                self.optimizer.step()

        return self.model

    def train_model_craig_subset(self,optimized_weights,selected_samples_indices):
        
        #create a subset of the train dataset with the selected samples
        craig_subset = Subset(self.train_dataset,selected_samples_indices)
        craig_subset_train_loader = DataLoader(craig_subset,batch_size=self.batch_size,shuffle=False)

      
        # Map optimized weights (length = #selected samples) to full
        # dataset index space (len(train_dataset)) so we can safely index by dataset indices
        # coming from the DataLoader.
      
        selected_indices_list = [int(idx) for idx in selected_samples_indices]
        selected_indices_tensor = torch.tensor(
            selected_indices_list, dtype=torch.long, device=self.device
        )

        # create a weight vector over the full training set and fill in weights
        full_sample_weights = torch.zeros(
            len(self.train_dataset),
            dtype=optimized_weights.dtype,
            device=self.device,
        )
        full_sample_weights[selected_indices_tensor] = optimized_weights.to(self.device)

        #move the model to the device
        self.model.train()
        self.model.to(self.device)

        losses_craig_train = []
        for epoch in range(self.num_epochs):
            #iterate over the data loader
            for indicies, inputs, labels in craig_subset_train_loader:

                # dataset indices for this batch
                batch_indices = indicies.to(self.device)

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                #(n_samples,)   reduction='none' to get the loss for each sample in the batch
                loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)

                #loss weighted by the optimized weights [batch_start:batch_end] for each sample in the batch
                # here we index into the full_sample_weights tensor using
                # dataset indices from this batch
                batch_weights = full_sample_weights[batch_indices]  #(n_samples,)
                loss = loss * batch_weights  #(n_samples,)
                loss.backward()
                self.optimizer.step()

                # store mean loss for logging
                losses_craig_train.append(loss.mean().item())

        with open('losses_craig_train.csv', 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(losses_craig_train)

        return self.model
    
    def craig_train(self):
        #train num_epochs epochs on selected samples then reselect: do that n_reselections times --> total #epochs = num_epochs*n_reselections

        # validation loader (
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        for epoch in range(self.num_epochs*self.n_reselections):

            #add epochs spent in craig train to current epoch            
            epoch = epoch + self.num_epochs
            

            # select coreset samples and their optimized weights
            selected_samples_indices, optimized_weights = self.select_samples_craig()

            # train on current craig coreset for self.num_epochs epochs
            self.model = self.train_model_craig_subset(optimized_weights, selected_samples_indices)

            
            #evaluate the model
            self.model.eval()
            with torch.no_grad():
                mean_loss_epoch = 0
                for indices, inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)  
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    mean_loss_epoch += loss.item()
                mean_loss_epoch /= len(val_loader)
            
            with open("loss_on_validation_set.csv", "a+") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, mean_loss_epoch])


        return self.model



    def select_samples_craig(self):

    ############################################################################################################################
    #Step 1: FULL GRADIENT CALCULATION (sum of all gradients of all samples)
    ############################################################################################################################

        # Ensure model is on device for gradient computations
        self.model.to(self.device)

        # data loader over the full training set
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        full_grad_sum = None  # will be initialized after first batch
        
        for indices, inputs, labels in train_loader:
            # Important: avoid accumulating parameter grads across batches
            self.model.zero_grad(set_to_none=True)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            batch_size = inputs.size(0)

            # Forward pass
            pred = self.model(inputs)
            loss = nn.CrossEntropyLoss()(pred, labels)
            loss.backward()

            grad_vec = torch.cat(
                [p.grad.view(-1) for p in self.model.parameters() if p.grad is not None]
            ).detach().clone()

            if full_grad_sum is None:
                full_grad_sum = torch.zeros_like(grad_vec)

            full_grad_sum += grad_vec
    

        print("full gradient sum shape: ", full_grad_sum.shape)
        print("full gradient sum size: ", full_grad_sum.numel() * full_grad_sum.element_size() / 1024 / 1024, "MB")

    ############################################################################################################################
    #Step 2: CORESET SELECTION
    ############################################################################################################################

        residual_grad = full_grad_sum.clone()

        selected_sample_indices: list[int] = []
        selected_sample_grads = []
        selected_sample_indices_set: set[int] = set()

        # Select exactly n_samples, one greedy choice per iteration.
        for t in range(self.n_samples):
            best_aligned_score = None
            best_aligned_index = None
            best_aligned_grad_vec = None

            # iterate over the data loader batch by batch
            for indices, inputs, labels in train_loader:
                self.model.zero_grad(set_to_none=True)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                batch_size = inputs.size(0)

                # Forward pass
                pred = self.model(inputs)
                per_sample_loss = nn.CrossEntropyLoss(reduction='none')(pred, labels)

                # model params 
                params = [p for p in self.model.parameters() if p.requires_grad]

                # Build candidate positions (skip already-selected samples)
                candidate_positions = []
                candidate_indices = []

                for i in range(batch_size):
                    idx_i = int(indices[i].item())
                    if idx_i in selected_sample_indices_set:
                        continue
                    candidate_positions.append(i)
                    candidate_indices.append(idx_i)

                # Evaluate alignment for candidates in this batch
                for j, i in enumerate(candidate_positions):
                    retain_graph = j < (len(candidate_positions) - 1)
                    grad_i = torch.autograd.grad(
                        per_sample_loss[i],
                        params,
                        retain_graph=retain_graph,
                        create_graph=False,
                    )
                    grad_i_vec = torch.cat([g.view(-1) for g in grad_i])

                    # dot product of vectors of size (n_parameters,)!!!!!!!!!!!!
                    alignment_score_i = torch.dot(grad_i_vec, residual_grad)
                    if (best_aligned_score is None) or (alignment_score_i > best_aligned_score):
                        best_aligned_score = alignment_score_i
                        best_aligned_index = int(indices[i].item())
                        best_aligned_grad_vec = grad_i_vec.detach()

            selected_sample_indices.append(best_aligned_index)
            selected_sample_indices_set.add(best_aligned_index)
            selected_sample_grads.append(best_aligned_grad_vec)

            # Convert selected gradients list to matrix G_S (m, d)
            selected_sample_grads_matrix = torch.stack(selected_sample_grads)

            # Solve for weights: (G_S G_S^T) gammas = G_S G_full
            A = selected_sample_grads_matrix @ selected_sample_grads_matrix.T  # (m, m)
            b = selected_sample_grads_matrix @ full_grad_sum                  # (m,)

            #  ridge for numerical stability 
            ridge = 1e-8 * torch.eye(A.size(0), device=A.device, dtype=A.dtype)
            optimized_weights = torch.linalg.solve(A + ridge, b)              # (m,)

            # Update residual: G_full - G_S^T * gammas
            residual_grad = full_grad_sum - selected_sample_grads_matrix.T @ optimized_weights

        return selected_sample_indices,optimized_weights


    def run_craig(self):
        self.model = self.train_model_warmup()
        self.model, loss_list_total = self.craig_train()
        return self.model, loss_list_total


if __name__ == "__main__":

    """
    craig training experiment on mhist dataset an resnet18
    """


    #read the config file
    config = get_config(config_path='./config.yml')

    #get the train and val datasets torchvision.datasets.ImageFolder + indices
    train_dataset, val_dataset = get_torch_datasets(batch_size=config['batch_size'],config_path='./config.yml')

    # determine number of classes from the ImageFolder dataset
    num_classes = len(train_dataset.classes)

    #initialize the model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # match num classes of the dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    craig_experiment = craig(
        model,
        optimizer,
        train_dataset,
        val_dataset,
        num_epochs_warmup=1,   # set as needed
        num_epochs=10,
        n_samples=50,
        n_reselections=10,
    )
    craig_experiment.run_craig()