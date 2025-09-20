import torch
import os
import PIL.Image as Image
import glob
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

IMAGE_SIZE = 128
BATCH_SIZE = 64
NUM_WORKERS = 3
DATA_PATH = './data/hotdog_nothotdog'

class Hotdog_NotHotdog(torch.utils.data.Dataset):
    def __init__(self, train, transform):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(DATA_PATH, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y

def make_datasets_and_dataloaders(transform=None):
    if not transform:
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    trainset = Hotdog_NotHotdog(train=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    testset = Hotdog_NotHotdog(train=False, transform=transform)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return trainset, testset, train_loader, test_loader