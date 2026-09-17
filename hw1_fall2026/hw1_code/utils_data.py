import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from PIL import Image
import os



class YaleB(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(YaleB, self).__init__(root, transform=transform, target_transform=target_transform)

        self.data = []
        self.targets = []
        root = root + "/CroppedYale"

        # Loop through the directory of each individual
        for i, person_dir in enumerate(sorted(os.listdir(root))):
            person_path = os.path.join(root, person_dir)
            if os.path.isdir(person_path):
                # Loop through each image of an individual
                for img_name in sorted(os.listdir(person_path)):
                    img_path = os.path.join(person_path, img_name)
                    if img_path.endswith(".pgm"):
                        img = Image.open(img_path)
                        self.data.append(img)
                        self.targets.append(i)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    
    
def load_dataset(name, root='./data/'):
    if name == 'MNIST':
        # Load MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        trainset = torchvision.datasets.MNIST(root=root, train=True, transform=transform, download=True)
    elif name == 'YALEB':
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.view(-1)) 
        ])
        trainset = YaleB(root=root, transform=transform)
    else:
        raise NameError(f'Dataset not found: {name}')
                        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=60000, shuffle=True)
    X_all, y_all = next(iter(trainloader))
    return X_all, y_all
