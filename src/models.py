import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.data import IMAGE_SIZE

class BaselineNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, 
            conv_configs=[(16, 3), (16, 3),  
                (32, 3), (32, 3),  
                (64, 3), (64, 3),  
                (128, 3)], # channels,kernel_size
        ):
        super().__init__()
    
        self.input_shape = (3, IMAGE_SIZE, IMAGE_SIZE)
        
        conv_blocks = []
        in_channels = self.input_shape[0]
        for out_channels, kernel_size in conv_configs:
            conv_blocks += self._make_conv_block(in_channels, out_channels, kernel_size)
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_blocks)
        
        flatten_dim = self._get_conv_output(self.input_shape)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
        )
    
    def _make_conv_block(self, in_channels, out_channels, kernel_size):
        """
        Conv block: Conv2d -> BatchNorm -> ReLU -> MaxPool
        """
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        ]
        return layers
    
    def _get_conv_output(self, shape):
        """Run a dummy tensor through conv layers to get output size."""
        dummy = torch.zeros(1, *shape)
        out = self.conv(dummy)
        return int(torch.prod(torch.tensor(out.shape[1:])))  # C*H*W
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class EfficientNet(nn.Module):
    def __init__(self, baseline_level: int = 0, freeze_backbone: bool = True):
        super().__init__()

        if not (0 <= baseline_level <= 7):
            raise ValueError("baseline_level must be between 0 and 7 (B0-B7).")

        model_fn = getattr(torchvision.models, f"efficientnet_b{baseline_level}")
        weights_cls = getattr(torchvision.models, f"EfficientNet_B{baseline_level}_Weights")
        weights = weights_cls.DEFAULT

        backbone = model_fn(weights=weights)

        self.transforms = weights.transforms()
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        in_features = backbone.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 1)
        )

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_transforms(self):
        return self.transforms