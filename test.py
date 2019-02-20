
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
from PIL import Image
import models
import utils
import os
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='directory of the dataset')
    parser.add_argument('--batchSize', type=int, default=1, help='batch size')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=128, help='size of squared img (crop)')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--genG', type=str, default='output/netG.pth', help='generator checkpoint file x->y')
    parser.add_argument('--genF', type=str, default='output/netF.pth', help='generator checkpoint file y->x')
    opt = parser.parse_args()
    print(opt)
    
    
    G = models.Generator(opt.input_nc, opt.output_nc)
    F = models.Generator(opt.input_nc, opt.output_nc)
    
    if opt.cuda:
        G.cuda()
        F.cuda()
        
    # Load state dicts
    G.load_state_dict(torch.load(opt.genG))
    F.load_state_dict(torch.load(opt.genF))
    
    G.eval()
    F.eval()
    
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_x = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_y = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    
    transformList = [ transforms.Resize(int(opt.size), Image.BICUBIC),
                     transforms.RandomCrop(opt.size),
            transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    
    dataset = DataLoader(utils.LoadDataset(opt.dataroot, transformList=transformList, mode='test'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu) 

    if not os.path.exists('output/x'):
        os.makedirs('output/x')
    if not os.path.exists('output/y'):
        os.makedirs('output/y')
        
        
    for i,batch in enumerate(dataset):
        currentBatch_x = Variable(input_x.copy_(batch['x']))
        currentBatch_y = Variable(input_y.copy_(batch['y'])) 
        
        # Generate output
        fake_y = 0.5*(G(currentBatch_x).data + 1.0)
        fake_x = 0.5*(F(currentBatch_y).data + 1.0)
    
        # Save image files
        save_image(fake_x, 'output/x/%04d.png' % (i+1))
        save_image(fake_y, 'output/y/%04d.png' % (i+1))
    
        sys.stdout.write('\rGenerated %04d of %04d' % (i+1, len(dataset)))
    
    sys.stdout.write('\n')        
        