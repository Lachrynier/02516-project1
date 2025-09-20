import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        # Convolutional block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Convolutional block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Convolutional block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 128)  # depends on input image size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)  # binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # block 1
        x = self.pool(F.relu(self.conv2(x)))  # block 2
        x = self.pool(F.relu(self.conv3(x)))  # block 3
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, 
                 input_shape=(3, 128, 128), 
                 conv_configs=[(16, 5), (32, 3)],
                 num_classes=10, 
                 conv_p=0.0,    # usually 0.0 or small, e.g., 0.1 for deep conv blocks
                 fc_p=0.5):     # stronger dropout in FC layers
        super().__init__()
        
        conv_blocks = []
        in_channels = input_shape[0]
        for i, (out_channels, kernel_size) in enumerate(conv_configs):
            # optional: add dropout only in deeper conv layers
            apply_dropout = (i == len(conv_configs)-1 and conv_p > 0)
            conv_blocks += self._make_conv_block(in_channels, out_channels, kernel_size, apply_dropout, conv_p)
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_blocks)
        
        flatten_dim = self._get_conv_output(input_shape)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 500),
            nn.ReLU(),
            nn.Dropout(p=fc_p),   # effective place for dropout
            nn.Linear(500, 1),
        )
    
    def _make_conv_block(self, in_channels, out_channels, kernel_size, apply_dropout, p):
        """One conv block: Conv2d + BN + ReLU + MaxPool (+ optional Dropout)"""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ]
        if apply_dropout:
            layers.append(nn.Dropout(p=p))  # only for deeper conv blocks if enabled
        return layers
    
    def _get_conv_output(self, shape):
        """Run a dummy tensor through conv layers to get output size."""
        dummy = torch.zeros(1, *shape)  # batch size = 1
        out = self.conv(dummy)
        return int(torch.prod(torch.tensor(out.shape[1:])))  # C*H*W
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
