import torchvision

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Subset

class ReverseIntensity:
    def __call__(self, x):
        return 1 - x

def main(path):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),
            ReverseIntensity(),
            transforms.Resize(32),
            # transforms.Pad(32, fill=255, padding_mode='constant'),
            # transforms.RandomRotation(45),
            # transforms.CenterCrop(32),
            transforms.RandomAffine(degrees=30, translate=(0.5, 0.5), scale=(0.25, 1), shear=(-30, 30, -30, 30)),            
    ])
    dataset = ImageFolder(path, transform=transform)
    train_loader = DataLoader(dataset, batch_size=800, shuffle=True)

    inputs_batch, labels_batch = next(iter(train_loader))
    grid = torchvision.utils.make_grid(inputs_batch, nrow=40, pad_value=1)
    torchvision.utils.save_image(grid, 'preview.png')

if __name__ == '__main__':
    path = '/home/teerawat.c/projects/handwritten-onnx-js/models/data'
    main(path)