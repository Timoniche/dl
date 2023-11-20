import torch

from torch import nn

import torchvision.transforms as transforms

from lab1.inception_v3_impl import InceptionV3


class InceptionModelV3(nn.Module):
    def __init__(
            self,
            num_classes,
            model_pretrained_path=None,
            classification_layer_path=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.inception_v3 = InceptionV3(num_classes=num_classes)

        if model_pretrained_path is not None:
            self.inception_v3.load_state_dict(torch.load(model_pretrained_path))

        self.inception_v3.fc = nn.Linear(in_features=2048, out_features=num_classes)

        if classification_layer_path is not None:
            self.inception_v3.fc.load_state_dict(torch.load(classification_layer_path))

        # self.freeze()

    def forward(self, x):
        x = self.normalize(x)
        x = self.inception_v3(x)

        if self.training:
            return x.logits

        return x

    def freeze(self):
        for param in self.inception_v3.parameters():
            param.requires_grad = False

        for param in self.inception_v3.fc.parameters():
            param.requires_grad = True

    def save_classificator_layer(self, path='classification_layer.pth'):
        torch.save(self.inception_v3.fc.state_dict(), path)
