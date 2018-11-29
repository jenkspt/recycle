import numpy as np
import time
import copy
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from transforms import RandomErasing

from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_class_weights():
    """ Account for class imbalance """
    train_weights = torch.Tensor([288.,242.,222.])
    train_weights = 1 - train_weights/train_weights.sum()

    valid_weights = torch.Tensor([215.,171.,88.])
    valid_weights = 1 - valid_weights/valid_weights.sum()

    class_weights = {
            'train':train_weights.to(DEVICE), 'valid':valid_weights.to(DEVICE)}
    return class_weights

def train_model(model, optimizer, scheduler, dataloaders, num_epochs=25):


    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = float('-inf')

    class_weights = get_class_weights()

    for epoch in range(1,num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}\n' + '-'*10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            confusion_matrix = torch.zeros(3, 3)

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = F.cross_entropy(outputs, labels, 
                            weight=class_weights[phase])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # Update confusion matrix
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            cls_acc = confusion_matrix.diag()/confusion_matrix.sum(1)
            avg_acc = cls_acc.mean()

            print(f'{phase} Loss: {epoch_loss:.4f} Avg Acc: {avg_acc*100:.2f}%',end='')
            classes = dataloaders[phase].dataset.classes
            strings = (f'{cls}: {acc*100:.2f}%' for cls,acc in zip(classes, cls_acc))
            print('\t('+' '.join(strings) + ')')

            # deep copy the model
            if phase == 'valid' and avg_acc > best_acc:
                best_acc = avg_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def get_dataloaders(batch_size=64):

    train_transform = transforms.Compose([
        transforms.RandomRotation((0, 360), resample=Image.BILINEAR),
        transforms.CenterCrop(447),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        RandomErasing(mean=[0, 0, 0])
        ])

    valid_transform = transforms.Compose([
        transforms.CenterCrop(447),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    trainset = datasets.ImageFolder(
            'data/recycle_classify/train', train_transform)

    validset = datasets.ImageFolder(
            'data/recycle_classify/valid', valid_transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, 
            shuffle=True, num_workers=8)
    validloader = DataLoader(validset, batch_size=batch_size, 
            shuffle=False, num_workers=8)

    dataloaders = {'train': trainloader, 'valid':validloader}
    return dataloaders

def get_model(pretrained=False):
    model = models.resnet50(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(),
        nn.Linear(num_ftrs, 3)
        )
    return model

if __name__ == "__main__":

    dataloaders = get_dataloaders()

    model = get_model(True)
    #model.load_state_dict(torch.load('classify/model.pt'))
    model = model.to(DEVICE)

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.90)

    # Decay LR by a factor of `gamma` every `step_size` epochs
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=14, gamma=0.5)

    model = train_model(
        model, 
        optimizer, 
        lr_scheduler,
        dataloaders,
        num_epochs=35)
    
    model_num = 6
    torch.save(model.state_dict(), f'classify/saved_models/model_{model_num}.pt')
