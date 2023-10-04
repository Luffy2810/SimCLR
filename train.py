from src.dataset.dataloader import get_mutated_dataloader
from src.model.ResnetSimCLR import make_model
from src.model.SIMCLR_LOSS import SimCLR_Loss
from src.model.LARS import LARS
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm







def get_mean_of_list(L):
    return sum(L) / len(L)

def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader_training_dataset_mutated = get_mutated_dataloader()

    resnet=make_model().to(device)
    losses_train = []
    num_epochs = 200


    optimizer = LARS(
        [params for params in resnet.parameters() if params.requires_grad],
        lr=0.2,
        weight_decay=1e-6,
        exclude_from_weight_decay=["batch_normalization", "bias"],
    )


    warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0, verbose = True)

 
    mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, eta_min=0.05, last_epoch=-1, verbose = True)

    
    criterion = SimCLR_Loss(batch_size = 128, temperature = 0.5)
    if not os.path.exists('results'):
        os.makedirs('results')

    if(os.path.isfile("results/model.pth")):
        resnet.load_state_dict(torch.load("results/model.pth"))
        optimizer.load_state_dict(torch.load("results/optimizer.pth"))
        temp = np.load("results/lossesfile.npz")
        losses_train = list(temp['arr_0'])
        resnet.train()

    for epoch in range(num_epochs):

        epoch_losses_train = []

        for (_, sample_batched) in enumerate(tqdm(dataloader_training_dataset_mutated)):

            optimizer.zero_grad()
            x1 = sample_batched['image1']
            x2 = sample_batched['image2']

            x1 = x1.to(device).float()
            x2 = x2.to(device).float()

            y1 = resnet(x1)
            y2 = resnet(x2)

            loss = criterion(y1, y2)
            loss.backward()
            if epoch < 10:
                warmupscheduler.step()
            if epoch >= 10:
                mainscheduler.step()

            epoch_losses_train.append(loss.cpu().data.item())
            optimizer.step()

        losses_train.append(get_mean_of_list(epoch_losses_train))
        fig = plt.figure(figsize=(10, 10))
        sns.set_style('darkgrid')
        plt.plot(losses_train)
        plt.legend(['Training Losses'])
        plt.savefig('losses.png')
        plt.close()
        torch.save(resnet.state_dict(), 'results/model.pth')
        torch.save(optimizer.state_dict(), 'results/optimizer.pth')
        np.savez("results/lossesfile", np.array(losses_train))

if __name__=="__main__":
    train()