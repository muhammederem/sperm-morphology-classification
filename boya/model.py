import torch
import torch.nn as nn
from torchvision import models


class BilinearCNN(nn.Module):
    def __init__(self, base_model, num_classes):
        super(BilinearCNN, self).__init__()
        # ResNet50’den son iki katmanı at (avgpool ve fc)
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.fc_dim = 2048  # ResNet50'de son katmanın kanal sayısı
        self.classifier = nn.Linear(self.fc_dim * self.fc_dim, num_classes)

    def bilinear_pooling(self, x):
        # x: (batch, channels, height, width)
        batch_size, ch, h, w = x.size()
        x = x.view(batch_size, ch, h * w)
        x = torch.bmm(x, x.transpose(1, 2)) / (h * w)  # Bilinear outer product
        x = x.view(batch_size, -1)
        x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10)  # Signed sqrt
        x = nn.functional.normalize(x)  # L2 norm
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.bilinear_pooling(x)
        x = self.classifier(x)
        return x


def create_model(num_classes):
    base_model = models.resnet50(pretrained=True)
    model = BilinearCNN(base_model, num_classes)
    return model
