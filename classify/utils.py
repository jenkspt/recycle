from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transforms import RandomErasing
from torchvision import datasets, models, transforms
from torchvision.transforms import ToPILImage

import trainer
from visualbackprop import ResnetVisualizer

import matplotlib.pyplot as plt
from matplotlib import colors

def rescale_to(arr, low=0, high=255):
    arr += low - arr.min()
    arr /= arr.max()
    arr *= high
    return arr

def display_topk(model, dataset,
        rows=3, cols=3, largest=True, color=False, figsize=(12,6)):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    loader = DataLoader(dataset, batch_size=64)
    label_probs_lst = []
    
    with torch.set_grad_enabled(False):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = F.softmax(logits, -1)
            _, preds = torch.max(probs, 1)
            # Probability assigned to the correct label
            label_probs_lst += [probs[range(len(probs)), labels].cpu()]

        label_probs = torch.cat(label_probs_lst, 0)

        label_probs, args = torch.topk(
                label_probs, k=rows*cols, largest=largest)

        # Batch of best or worst images
        images = torch.stack([dataset[i][0] for i in args],0)
        labels = [dataset[i][1] for i in args]
        # Run images through the feature visualizer
        model_vis = ResnetVisualizer(model.to("cpu")).eval()
        probs, vis = model_vis(images)
        _, preds = torch.max(probs, 1)
        
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        
        zipped = zip(preds, labels, label_probs, images, vis, axs.ravel())
        for pred, label, label_prob, image, feat, ax in zipped:
            if not color:
                image[...] = image.mean(0, keepdim=True)
            # Convert torch tensors to PIL images
            image = rescale_to(image, 0, 255).round().to(torch.uint8)
            image = ToPILImage()(image)

            feat = rescale_to(feat, 0, 255).round().to(torch.uint8)
            feat = ToPILImage()(feat).point(lambda px: np.abs(px-255))
            
            # Overlay feature map on image
            red = Image.new('RGB', image.size, (255,0,0))
            mask = Image.new('RGBA',image.size,(0,0,0,123))
            mask.putalpha(feat)

            out = Image.composite(image, red, mask)

            #image, feat = to_img(image), to_img(feat)
            ax.imshow(out, interpolation='bicubic')
            ax.axis('off')
            _pred = validset.classes[pred]
            _label = validset.classes[label]
            ax.set_title(f'Pred: {_pred}, Actual: {_label}, \nProb: {label_prob:.2f}')

        return axs
        

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = trainer.get_model(False)

    model.load_state_dict(torch.load('classify/saved_models/model_6.pt'))
    model = model.to(device)

    valid_transform = transforms.Compose([
        transforms.CenterCrop(447),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    trainset = datasets.ImageFolder(
        'data/recycle_classify/train', valid_transform)

    validset = datasets.ImageFolder(
        'data/recycle_classify/valid', valid_transform)

    display_topk(model, trainset, rows=2, cols=4, largest=False)
    plt.show()
