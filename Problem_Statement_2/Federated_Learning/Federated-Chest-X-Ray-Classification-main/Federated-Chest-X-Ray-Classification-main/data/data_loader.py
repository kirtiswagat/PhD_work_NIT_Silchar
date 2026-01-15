import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ChestXRayDataset(Dataset):
    def __init__(self, client_id, split, transform=None):
        samples = []
        for label, cname in enumerate(['NORMAL', 'PNEUMONIA']):
            dir_ = f'data/partitions/client_{client_id}/{split}/{cname}'
            if os.path.isdir(dir_):
                samples += [(os.path.join(dir_, f), label) for f in os.listdir(dir_)]
        self.samples = samples
        self.transform = transform or transforms.Compose([
            transforms.Resize((128,128)), 
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label

def get_client_loader(client_id, split, batch_size=16):
    dataset = ChestXRayDataset(client_id, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# test loader
if __name__ == '__main__':
    loader = get_client_loader(1, "train", batch_size=8)
    for imgs, labels in loader:
        print(imgs.shape, labels)
        break  # Only show one batch