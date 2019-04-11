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
from yoloLoss_mimic import yoloLossMimic
from mimic_net import vgg16,vgg16XNOR
import util

def save_checkpoint(state, is_best, filename='./experiment/vgg16mix/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './experiment/vgg16mix/model_best.pth.tar')
      
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch <5:
        lr =  1e-5
    if epoch >=5 and epoch < 80:
        lr = 1e-4
    if epoch >= 80 and  epoch < 110:
        lr = 1e-5
    if epoch >=110:
        lr = 1e-5
    print 'Learning rate:', lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, student_model, teacher_model,gt_criterion,mm_criterion,optimizer, epoch):
    total_loss = 0.
    total_gt_loss = 0.
    total_mm_loss = 0.
    student_model.train()
    teacher_model.eval()
    for i, (input, target) in enumerate(train_loader):

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        input_var,target_var = input_var.cuda(),target_var.cuda()
        
        '''Weight Binarization.'''
        bin_op.binarization()
        
        '''Forward propagation and compute yolo loss.'''
        student_feature, student_output = student_model(input_var)
        #teacher_feature, teacher_output  = teacher_model(input_var)
        
        gt_loss = gt_criterion(student_output,target_var)
        #mm_loss = mm_criterion(student_feature,teacher_feature)/args.batch_size
        loss = gt_loss #+ 2*mm_loss


        total_loss += loss.data.item()
        total_gt_loss += gt_loss.data.item()
        #total_mm_loss += mm_loss.data.item()

        '''Computer gradient.'''
        optimizer.zero_grad()
        loss.backward()

        '''Restore binarized weight to full precision and update.'''
        
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        
        optimizer.step()
        
        '''#Print losses etc.'''
        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f, gt_loss:%.4f, mm_loss:%.8f' 
            %(epoch+1, args.epochs, i+1, len(train_loader), loss.data.item(), total_loss / (i+1),total_gt_loss/(i+1),total_mm_loss/(i+1)))
            
            with open("./log/log.txt", "a") as text_file:
                text_file.write('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f \n, gt_loss:%.4f, mm_loss:%.8f' 
                %(epoch+1, args.epochs, i+1, len(train_loader), loss.data.item(), total_loss / (i+1),total_gt_loss/(i+1),total_mm_loss/(i+1)))
    gc.collect()

def validate(test_loader,student_model,teacher_model, gt_criterion):
    validate_loss = 0.0
    '''Weight Binarization.'''
    student_model.eval()
    teacher_model.eval()
    bin_op.binarization()
    for i, (input, target) in enumerate(test_loader):
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        input_var,target_var = input_var.cuda(),target_var.cuda()
        student_feature, student_output = student_model(input_var)
        #teacher_feature, teacher_output = teacher_model(input_var)

        gt_loss = gt_criterion(student_output,target_var)
        loss = gt_loss
        validate_loss += loss.data.item()

    validate_loss /= len(test_loader)
    print('validation loss %.5f' % validate_loss)
    with open("./log/log.txt", "a") as text_file:
        text_file.write('validation loss %.5f \n' % validate_loss)
   
    bin_op.restore()

    return validate_loss

def main():
    
    '''Parse argument.'''
    parser = argparse.ArgumentParser(description = 'Pytorch XNOR-YOLO Training')
    
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
    parser.add_argument('--l', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    default=False, help='use pre-trained model')
    parser.add_argument('--mixnet', dest='mixnet', action='store_true',
                    default=False, help='use mixnet model')
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
        list_file=['./meta/voc2007.txt', './meta/voc2012.txt'], train=True, transform=[transforms.ToTensor()])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    

    test_dataset = yoloDataset(root ='/mnt/lustre/share/DSK/datasets/VOC07+12/JPEGImages/',
        list_file='./meta/voc2007test.txt',train=False, transform=[transforms.ToTensor()])
    
    test_loader = DataLoader(
        test_dataset,batch_size=args.batch_size, shuffle=False, num_workers=4)


    '''Create model.'''
    teacher_model = vgg16(pretrained = False)
    student_model = vgg16XNOR(pretrained = False)

    teacher_model = torch.nn.DataParallel(teacher_model)
    student_model.features = torch.nn.DataParallel(student_model.features)
    teacher_model.cuda() 
    student_model.cuda() 
  
 
    '''Define loss functin i.e. YoloLoss and optimizer i.e. ADAM'''
    gt_criterion = yoloLoss(7,2,5,0.5)
    mm_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(student_model.parameters(), args.lr,
                                weight_decay = args.l)
    
    '''weight initialization'''
    for m in student_model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            c = float(m.weight.data[0].nelement())
            m.weight.data = m.weight.data.normal_(0, 2.0/c)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data = m.weight.data.zero_().add(1.0)
            m.bias.data = m.bias.data.zero_()

    '''weight loading'''
    teacher_model.load_state_dict(torch.load('./experiment/vgg16fp/checkpoint.pth'))
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            student_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
  
    
    print student_model, teacher_model
    
    '''Define binarization operator.'''
    global bin_op
    bin_range = [1,11]
    bin_op = util.BinOp(student_model,bin_range)
    
    best_loss = 100

    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)
        '''Train& validate for one epoch.'''
        train(train_loader, student_model, teacher_model, gt_criterion, mm_criterion, optimizer, epoch)
        val_loss = validate(test_loader, student_model,teacher_model, gt_criterion)
        
        is_best = val_loss < best_loss 
        best_loss = min(val_loss,best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': student_model.state_dict(),
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
     
if __name__ == '__main__':
    main()




