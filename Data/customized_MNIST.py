import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_ 

def MNIST_loaders(train_batch_size=50000, test_batch_size=1):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader



class CustomMNISTDataset(Dataset):
    def __init__(self, root='./loaddata/', train=True):
        """
        Args:
            root (str): Root directory of the MNIST dataset.
            train (bool): True for the training dataset, False for the test dataset.
            custom_transform (callable, optional): A function/transform to apply to the data.
        """ 

        transform = Compose([
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,)),
                    Lambda(lambda x: torch.flatten(x))])

        self.mnist_dataset = MNIST(
            root=root,
            train=train,
            transform=transform,  # Apply custom transformations if provided
            download=True, )

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        x, y = self.mnist_dataset[idx]  
        return x,y
 

def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()

if __name__ == "__main__":

    batch_size = 2
    MNIST_aug    = CustomMNISTDataset( train=True)
    train_loader = DataLoader(MNIST_aug,batch_size=1, shuffle=True) 

    pbar = tqdm(train_loader)

    for i, data in enumerate(pbar): 
        x,y = data
