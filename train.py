import argparse
import itertools
import torch 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import models
import utils
import sys
import datetime
import time
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import random

#------------------------- HELPER FUNC AND CLASS -----------------------------

def initWeights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

class LR_sched():
    def __init__(self, numEpochs, decayEpoch):
        assert ((numEpochs - decayEpoch) > 0), "ohh no, decay > number epochs"
        self.numEpochs = numEpochs
        self.decayEpoch = decayEpoch
    def step(self, currentEpoch):
        return 1.0 - max(0, currentEpoch - self.decayEpoch)/(self.numEpochs - self.decayEpoch)
#---------
        
#----------    
class ImageBuffer():
    def __init__(self, size=50):
        self.size = size
        self.bufferSize = 0
        self.buffer = []
    def pushPop(self, data):
        if self.size == 0:
            return data
        returnData = []
        for element in data:
            element = torch.unsqueeze(element.data, 0)
            if self.bufferSize < self.size:  
                self.bufferSize +=  1
                self.buffer.append(element)
                returnData.append(element)
            else:
                p = random.uniform(0, 1)
                if p > 0.5: 
                    random_id = random.randint(0, self.size - 1)  
                    tmp = self.buffer[random_id].clone()
                    returnData.append(tmp)
                    self.buffer[random_id] = element
                else:      
                    returnData.append(element)   
        return torch.cat(returnData, 0) 
#----------
class LossLogger():
    def __init__(self, numEpochs, numBatches):        
        self.numEpochs =numEpochs
        self.numBatches = numBatches
        self.losses = {}
        self.timeStart = time.time()
        self.timeBatchAvg = 0
            
    def log(self, currentEpoch, currentBatch, losses):        
        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] | ' % (currentEpoch, self.numEpochs, currentBatch, self.numBatches))        
        for lossName in losses:
            if lossName not in self.losses:
                self.losses[lossName] = []
                self.losses[lossName].append(losses[lossName].item())
            else:
                if len(self.losses[lossName]) < currentEpoch:
                    self.losses[lossName].append(losses[lossName].item())
                else:
                    self.losses[lossName][-1] += losses[lossName].item()                    
            sys.stdout.write('%s: %.4f | ' % (lossName, self.losses[lossName][-1]/currentBatch))
            if currentBatch % self.numBatches == 0 :
                self.losses[lossName][-1] *= 1./currentBatch                
            
        batchesDone =  (currentEpoch-1)*self.numBatches + currentBatch
        self.timeBatchAvg = (time.time() - self.timeStart)/float(batchesDone)
        batchesLeft = self.numEpochs*self.numBatches - batchesDone
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batchesLeft*self.timeBatchAvg)))
        
        if currentBatch % self.numBatches == 0 :
            sys.stdout.write('\n')
            
    def plot(self):
        for lossName in self.losses:
            plt.figure()
            plt.plot(range(len(self.losses[lossName])),self.losses[lossName])
            plt.title(lossName)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig('output/'+lossName+'.png')

    def save(self):
        df = pd.DataFrame.from_dict(self.losses)
        df.to_csv("output/losses.csv")

#------------------------------------------------------------------------------
if __name__ == '__main__':    
#    ---------------------- ARGS 
    parser = argparse.ArgumentParser()
#    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--numEpochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--batchSize', type=int, default=1, help='batch size')
    parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decayEpoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--lambdaCyc_x', type=float, default=10.0, help='lambda for cycle loss (x -> y -> x)')
    parser.add_argument('--lambdaCyc_y', type=float, default=10.0, help='lambda for cycle loss (y -> x -> y)')
    parser.add_argument('--lambdaIdentity', type=float, default=5.0, help='lambda for identity loss')
    parser.add_argument('--size', type=int, default=128, help='size of squared img to use (resize and crop)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--saveEpochFrq', type=int, default=25, help='frequency of saving checkpoints at the end of epochs')    
    parser.add_argument('--manualSeed', action='store_true', help='use manual seed')
    parser.add_argument('--seedNum', type=int, default=6, help='seed')  
    parser.add_argument('--imageBuffer', action='store_true', help='use an image buffer')  
    opt = parser.parse_args()
    print(opt)    
    
#    -------------------- VARS DEF 
    if opt.manualSeed:
        torch.cuda.manual_seed(opt.seedNum)
        torch.cuda.manual_seed_all(opt.seedNum)
#    nn
    G = models.Generator(opt.input_nc, opt.output_nc) #generator x->y
    F = models.Generator(opt.output_nc, opt.input_nc) #generator y->x
    D_x = models.Discriminator(opt.input_nc) #discriminator X
    D_y = models.Discriminator(opt.input_nc) #discriminator Y
        
    if opt.cuda:
        G.cuda()
        F.cuda()
        D_x.cuda()
        D_y.cuda()
     
    G.apply(initWeights)
    F.apply(initWeights)
    D_x.apply(initWeights)
    D_y.apply(initWeights)       

#    loss
    criterionGAN = torch.nn.MSELoss()
    criterionCycle = torch.nn.L1Loss()
    criterionIdentity = torch.nn.L1Loss()    

#   optim   
    optimizer_Genrators = torch.optim.Adam(itertools.chain(G.parameters(), F.parameters()),
                                    lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_x = torch.optim.Adam(D_x.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_y = torch.optim.Adam(D_y.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    lrScheduler_Genrators = torch.optim.lr_scheduler.LambdaLR(optimizer_Genrators, lr_lambda=LR_sched(opt.numEpochs, opt.decayEpoch).step)
    lrScheduler_D_x = torch.optim.lr_scheduler.LambdaLR(optimizer_D_x, lr_lambda=LR_sched(opt.numEpochs, opt.decayEpoch).step)
    lrScheduler_D_y = torch.optim.lr_scheduler.LambdaLR(optimizer_D_y, lr_lambda=LR_sched(opt.numEpochs, opt.decayEpoch).step)

    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_x = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_y = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    targetReal = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    targetFake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)
 
    if opt.imageBuffer:
        bufferFake_x = ImageBuffer()
        bufferFake_y = ImageBuffer()
    
#   ---------------------- LOAD DATA
    transformList = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
                    transforms.RandomCrop(opt.size), 
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    
    dataset = DataLoader(utils.LoadDataset(opt.dataroot, transformList=transformList), 
                            batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)        

    logger = LossLogger(opt.numEpochs, len(dataset))

#   --------------------- TRAIN    
    for epoch in range(1,opt.numEpochs+1):
        for i, batch in enumerate(dataset):
            currentBatch_x = Variable(input_x.copy_(batch['x']))
            currentBatch_y = Variable(input_y.copy_(batch['y'])) 
            
            
            fake_y = G(currentBatch_x) #G(x)
            fake_x = F(currentBatch_y) #F(y)
            
            #-------- Generators loss            
            optimizer_Genrators.zero_grad()
                        
            #lsgan loss
            lossGAN_G = criterionGAN(D_y(fake_y), targetReal)  # (D_y(G(x)) - 1)^2
            lossGAN_F = criterionGAN(D_x(fake_x), targetReal)  # (D_x(F(y)) - 1)^2
            
            #cycle loss
            recovered_x = F(fake_y) # F(G(x))
            recovered_y = G(fake_x) # G(F(y))
            lossCyc_x = criterionCycle(recovered_x, currentBatch_x) # | F(G(X)) - x |
            lossCyc_y = criterionCycle(recovered_y, currentBatch_y) # | G(F(y)) - y |
            lossCyc = lossCyc_x*opt.lambdaCyc_x + lossCyc_y*opt.lambdaCyc_y
                    
            #identity loss
            lossId_x = criterionIdentity(F(currentBatch_x), currentBatch_x) # | F(x) - x |
            lossId_y = criterionIdentity(G(currentBatch_y), currentBatch_y) # | G(y) - y |
            lossId = (lossId_x + lossId_y)*opt.lambdaIdentity
                        
            #total generators loss
            loss_Generators = lossGAN_G + lossGAN_F + lossCyc + lossId
            loss_Generators.backward()
            optimizer_Genrators.step()
            
            #-------- Discriminator loss
            #lsgan loss            
            optimizer_D_x.zero_grad()
            if opt.imageBuffer:
                lossGAN_D_x = (criterionGAN(D_x(currentBatch_x), targetReal) + criterionGAN(D_x(bufferFake_x.pushPop(fake_x).detach()), targetFake))*0.5 #(D_x(x)-1)^2 + (D_x(F(y)))^2
            else:
                lossGAN_D_x = (criterionGAN(D_x(currentBatch_x), targetReal) + criterionGAN(D_x(fake_x.detach()), targetFake))*0.5 #(D_x(x)-1)^2 + (D_x(F(y)))^2
            lossGAN_D_x.backward()
            optimizer_D_x.step()
            
            optimizer_D_y.zero_grad()
            if opt.imageBuffer:
                lossGAN_D_y = (criterionGAN(D_y(currentBatch_y), targetReal) + criterionGAN(D_y(bufferFake_y.pushPop(fake_y).detach()), targetFake))*0.5 #(D_y(y)-1)^2 + (D_y(G(x)))^2
            else:
                lossGAN_D_y = (criterionGAN(D_y(currentBatch_y), targetReal) + criterionGAN(D_y(fake_y.detach()), targetFake))*0.5 #(D_y(y)-1)^2 + (D_y(G(x)))^2
            lossGAN_D_y.backward()
            optimizer_D_y.step()         


            losses = {'loss_Gen': loss_Generators, 
                      'loss_Gen_identity': lossId, 
                      'loss_Gen_GAN': (lossGAN_G + lossGAN_F),
                        'loss_Gen_cycle': (lossCyc), 
                        'loss_Disc': (lossGAN_D_x + lossGAN_D_y)}
            logger.log(epoch, i+1, losses)
            
        lrScheduler_Genrators.step()
        lrScheduler_D_x.step()
        lrScheduler_D_y.step()
                
        # Save models
        if epoch % opt.saveEpochFrq == 0:
            label = '_ep'+str(epoch)
            torch.save(G.state_dict(), 'output/netG'+label+'.pth')
            torch.save(F.state_dict(), 'output/netF'+label+'.pth')
            torch.save(D_x.state_dict(), 'output/netD_x'+label+'.pth')
            torch.save(D_y.state_dict(), 'output/netD_y'+label+'.pth')    
            
        torch.save(G.state_dict(), 'output/netG.pth')
        torch.save(F.state_dict(), 'output/netF.pth')
        torch.save(D_x.state_dict(), 'output/netD_x.pth')
        torch.save(D_y.state_dict(), 'output/netD_y.pth')            
            
        logger.save()
        
    logger.plot()
            
            
            
            
            
