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
    if transform == "aug":
        # temporary transform to compute mean/std
        tmp_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        tmp_trainset = Hotdog_NotHotdog(train=True, transform=tmp_transform)
        mean, std = compute_mean_std(tmp_trainset)

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.GaussianBlur(kernel_size=3, sigma=(1e-4, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        ])

    elif transform is None:
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        test_transform = train_transform
    else:
        train_transform = transform
        test_transform = transform

    trainset = Hotdog_NotHotdog(train=True, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    testset = Hotdog_NotHotdog(train=False, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return trainset, testset, train_loader, test_loader

def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    mean = 0.0
    std = 0.0
    total = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total += batch_samples
    mean /= total
    std /= total
    return mean, std