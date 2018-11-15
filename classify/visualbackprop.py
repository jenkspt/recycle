import torch
import torch.nn as nn
from torch.nn.functional import conv_transpose2d
import torchvision

from torchvision.models.resnet import Bottleneck
from torchvision.models import resnet50


class ResnetVisualizer(nn.Module):

    def __init__(self, resnet):
        super(ResnetVisualizer, self).__init__()
        self.model = resnet

        for name, child in self.model.named_children():
            if 'layer' in name:
                setattr(self, name, LayerVisualizer(child))
        
        # For Deconv
        self.k7x7 = torch.ones((1,1,7,7))
        self.k3x3 = torch.ones((1,1,3,3))

    def forward(self, x):
        input = x
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        act1 = x.mean(1, keepdim=True)
        x = self.model.maxpool(x)

        x, vis1 = self.layer1(x)
        x, vis2 = self.layer2(x)
        x, vis3 = self.layer3(x)
        x, vis4 = self.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        
        vis = list(reversed([act1] + vis1 + vis2 + vis3 + vis4))
        prod = vis[0]
        for i in range(1, len(vis)):
            act = vis[i]
            if prod.shape != act.shape:
                prod = conv_transpose2d(
                        prod, self.k3x3, 
                        stride=2, padding=1, 
                        output_padding=1)
            #print(f'{i} prod:{tuple(prod.shape)}, act:{tuple(act.shape)}')
            prod *= act

        # Resize to input image
        prod = conv_transpose2d(
                prod, self.k7x7, stride=2, padding=3, output_padding=1)
        return x, prod #* input.mean(1, keepdim=True)

class LayerVisualizer(nn.Module):
    
    def __init__(self, layer):
        super(LayerVisualizer, self).__init__()
        self.layer = layer

        for name, child in self.layer.named_children():
            setattr(self, name, BottleneckVisualizer(child))

    def forward(self, x):

        vis=[]
        for name, child in self.layer.named_children():
            block = getattr(self, name)
            x, prod = block(x)
            vis += prod

        """
        vis = []   # Activations
        for block in self.layer.children():
            x = block(x)
            vis.append(x.mean(1,keepdim=True))   # Average channels
            print(f'LayerVis: {tuple(vis[-1].shape)}')
        """
        return x, vis

class BottleneckVisualizer(nn.Module):

    def __init__(self, block):
        super(BottleneckVisualizer, self).__init__()
        self.block = block
        self.k3x3 = torch.ones((1,1,3,3))
        self.k1x1 = torch.ones((1,1,1,1))

    def forward(self, x):
        vis = []

        residual = x

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        vis += [out.mean(1,keepdim=True)]


        out = self.block.conv2(out)
        out = self.block.bn2(out)
        out = self.block.relu(out)

        vis += [out.mean(1,keepdim=True)]


        out = self.block.conv3(out)
        out = self.block.bn3(out)

        if self.block.downsample is not None:
            residual = self.block.downsample(x)

        out += residual
        out = self.block.relu(out)
        vis += [out.mean(1,keepdim=True)]

        return out, [vis[-1]] #vis#[out.mean(1,keepdim=True)]

if __name__ == "__main__":
    model = resnet50(pretrained=True)
    model_vis = ResnetVisualizer(model)
