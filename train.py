import argparse
import os 
import shutil
import sys
import gc

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim
import torch.utils
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

from dataset import yoloDataset
from yoloLoss import yoloLoss
import model_list
from net import vgg16_bn
from mixnet import vgg16_mix
import util

def save_checkpoint(state, is_best, filename='./experiment/vgg16xnor/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './experiment/vgg16xnor/model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch <5:
        lr =  1e-5
    if epoch >=5 and epoch < 80:
        lr = 1e-4
    if epoch >= 80 and  epoch < 110:
        lr = 1e-5
    if epoch >=110:
        lr = 1e-6
    print 'Learning rate:', lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch):
    total_loss = 0.
    model.train()
    for i, (input, target) in enumerate(train_loader):

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        input_var,target_var = input_var.cuda(),target_var.cuda()
        

        if not args.pretrained:
            '''Weight Binarization.'''
            bin_op.binarization()
        
        '''Forward propagation and compute yolo loss.'''
        output = model(input_var)
        loss = criterion(output, target_var)
        total_loss += loss.data.item()

        '''Computer gradient.'''
        optimizer.zero_grad()
        loss.backward()

        '''Restore binarized weight to full precision and update.'''
        if not args.pretrained:
            bin_op.restore()
            bin_op.updateBinaryGradWeight()
        
        optimizer.step()
        
        '''#Print losses etc.'''
        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
            %(epoch+1, args.epochs, i+1, len(train_loader), loss.data.item(), total_loss / (i+1)))
            
            with open("./log/log.txt", "a") as text_file:
                text_file.write('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f \n' 
                %(epoch+1, args.epochs, i+1, len(train_loader), loss.data.item(), total_loss / (i+1)))
    gc.collect()

def validate(test_loader, model, criterion):
    validate_loss = 0.0
    '''Weight Binarization.'''
    model.eval()
    if not args.pretrained:
        bin_op.binarization()
    for i, (input, target) in enumerate(test_loader):
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        input_var,target_var = input_var.cuda(),target_var.cuda()
        output = model(input_var)
        loss = criterion(output,target_var)
        validate_loss += loss.data.item()

    validate_loss /= len(test_loader)
    print('validation loss %.5f' % validate_loss)
    with open("./log/log.txt", "a") as text_file:
        text_file.write('validation loss %.5f \n' % validate_loss)
    if not args.pretrained:
        bin_op.restore()

    return validate_loss

def main():
    
    '''Parse argument.'''
    parser = argparse.ArgumentParser(description = 'Pytorch XNOR-YOLO Training')
    
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
    parser.add_argument('--l', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    default=False, help='use pre-trained model')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
    global args
    args = parser.parse_args()

    '''Data loading module'''
    train_dataset = yoloDataset(root='/mnt/lustre/share/DSK/datasets/VOC07+12/JPEGImages/',
        list_file=['./meta/voc2007.txt'], train=True, transform=[transforms.ToTensor()])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    

    test_dataset = yoloDataset(root ='/mnt/lustre/share/DSK/datasets/VOC07+12/JPEGImages/',
        list_file='./meta/voc2007test.txt',train=False, transform=[transforms.ToTensor()])
    
    test_loader = DataLoader(
        test_dataset,batch_size=args.batch_size, shuffle=False, num_workers=4)


    '''Create model.'''
    if args.pretrained:
        print 'Loading a pretrained vgg16_bn model...'
        model = vgg16_bn(pretrained = False)
    
        pretrained_dict = torch.load('vgg16_bn-6c64b313.pth')
        model_dict = model.state_dict()
        pretrained_dict = pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                        (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    elif args.mixnet:
        print 'Loading a xnor&fp mixed model...'
        model,bin_range = vgg16_mix3()
    else:
        print 'Loading a binary vgg16_bn model...'
        model = model_list.vgg(pretrained = False)
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    
    '''Define loss functin i.e. YoloLoss and optimizer i.e. ADAM'''
    criterion = yoloLoss(7,2,5,0.5)
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay = args.l)
    '''weight initialization'''
    if not args.pretrained:
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                c = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, 2.0/c)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data = m.weight.data.zero_().add(1.0)
                m.bias.data = m.bias.data.zero_()


    best_loss = 100
    if not args.pretrained:
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_loss = checkpoint['best_loss']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                del checkpoint
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        if args.resume:
            model.load_state_dict(torch.load('./experiment/vgg16fp/checkpoint.pth'))

    
    print model
    '''Define binarization operator.'''
    
    global bin_op
    bin_op = util.BinOp(model,bin_range)
    
    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)
        '''Train& validate for one epoch.'''
        train(train_loader, model, criterion, optimizer, epoch)
        val_loss = validate(test_loader, model, criterion)
        
        is_best = val_loss < best_loss 
        best_loss = min(val_loss,best_loss)

        if not args.pretrained:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        else: 
            torch.save(model.state_dict(),'./experiment/vgg16fp/checkpoint.pth')
        
if __name__ == '__main__':
    main()




