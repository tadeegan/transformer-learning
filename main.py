#! /bin/python
import torch
# import numpy as np
import math
import mnist
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from typing import List
from tile_mnist import tile_mnist
from vit import ViT, MLP
from conv_net import Net as ConvNet
import wandb

# mnist.init()

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(28*28, 500),
            torch.nn.Linear(500, 10),
        ])

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for l in self.layers:
            x =  F.relu(l(x))
        output = F.log_softmax(x, dim=1)
        return output

def train(type='VIT'):

    x_train, t_train, x_test, t_test = mnist.load()


    wandb.init(project="vit-transformer")

    if type == 'VIT':
        model = ViT(input_channels=4*4, encoder_d=128, class_channels=10, grid_shape=(7,7))
    elif type == 'CONV':
        model = ConvNet()
    elif type == 'MLP':
        model = MLP()

    num_params = sum(param.numel() for param in model.parameters())
    print(f'#params: {num_params}')
    wandb.log({'model/parameters': num_params})

    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    learning_rate = 1e-3

    batch_size = 64
    epochs = 50
    step = 0

    patches_train = tile_mnist(imgs=x_train.reshape(-1,1,28,28), shape=(7,7))
    flat_patches_train = patches_train.reshape(patches_train.shape[0], patches_train.shape[1], -1)
    train_set = flat_patches_train if type == 'VIT' else x_train.reshape(-1,1,28,28)
    print(f'train_set.shape: {train_set.shape}')
    patches_test = tile_mnist(imgs=x_test.reshape(-1,1,28,28), shape=(7,7))
    flat_patches_test = patches_test.reshape(patches_test.shape[0], patches_test.shape[1], -1)
    test_set = flat_patches_test if type == 'VIT' else x_test.reshape(-1,1,28,28)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='max', verbose=True, patience=2)


    for e in range(epochs):
        model.cuda()
        model.train()
        num_batches = int(len(t_train) / batch_size)
        for b in tqdm(range(num_batches)):
            batch_start = batch_size * b
            x = train_set[batch_start:batch_start+batch_size]
            t = t_train[batch_start:batch_start+batch_size]
            x = torch.from_numpy(x).type(torch.FloatTensor) / 255.0 - 0.5
            x = x.cuda()

            y_pred = model(x)
            target = torch.Tensor(t).to(torch.int64).cuda()

            target_one_hot = F.one_hot(target, num_classes=10).to(torch.float32)

            loss = loss_fn(y_pred, target_one_hot)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

        print(f'loss: [{step}]: {loss.item()}')
        wandb.log({'loss': loss.item()})

        # model.cpu()
        model.eval()

        num_test_steps = 1000

        num_correct = 0
        
        for i in range(num_test_steps):
            x = torch.from_numpy(train_set[i:i+1]).type(torch.FloatTensor) / 255.0 - 0.5
            # x = x.unsqueeze(dim=-1)
            y = model(x.cuda()).to('cpu')
            output = np.argmax(y.detach().numpy(), axis=-1)
            # print(f'{t_train[i]} -> {output}')
            
            if t_train[i] == output:
                num_correct += 1

        accuracy = num_correct/ float(num_test_steps)
        wandb.log({'train/accuracy': accuracy})
        print(f'Accuracy (on train): {accuracy}')

        num_correct = 0
        for i in range(num_test_steps):
            x = torch.from_numpy(test_set[i:i+1]).type(torch.FloatTensor) / 255.0 - 0.5
            y = model(x)
            output = np.argmax(y.detach().numpy(), axis=-1)
            # print(f'{t_test[i]} -> {output}')

            if t_test[i] == output:
                num_correct += 1

        accuracy = num_correct/ float(num_test_steps)
        wandb.log({'validation/accuracy': accuracy})
        print(f'Accuracy (on test): {accuracy}')
        # import pdb; pdb.set_trace()
        scheduler.step(metrics=accuracy)
        wandb.log({'learning_rate': optimizer.param_groups[0]['lr']})


if __name__ == "__main__":
    train()