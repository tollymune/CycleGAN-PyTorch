
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
import glob
import os
from PIL import Image


class LoadDataset(Dataset):
    def __init__(self, dataRoot, transformList, mode='train'):
        self.transform = transforms.Compose(transformList)
        self.fileNames_x = sorted(glob.glob(os.path.join(dataRoot, '%s/x' % mode) + '/*.*')) 
        self.fileNames_y = sorted(glob.glob(os.path.join(dataRoot, '%s/y' % mode) + '/*.*'))
        self.mode = mode
    def __getitem__(self, index):
        item_x = self.transform(Image.open(self.fileNames_x[index % len(self.fileNames_x)]))
        if self.mode=='train':
            item_y = self.transform(Image.open(self.fileNames_y[random.randint(0, len(self.fileNames_y) - 1)])) #unaligned
        else:
            item_y = self.transform(Image.open(self.fileNames_y[index % len(self.fileNames_y)])) #aligned
        return {'x': item_x, 'y': item_y}
    
    def __len__(self):
        return max(len(self.fileNames_x), len(self.fileNames_y))  