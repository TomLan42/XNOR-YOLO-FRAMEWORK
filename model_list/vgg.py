import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['BinVGG16Detec', 'vgg','vggDetecFP']

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        input = input.sign()
        return input

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinConv2d(nn.Module): 
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0,
            Linear=False):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        else:
            self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        x = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            x = self.linear(x)
        x = self.relu(x)
        return x


class VGG16 (nn.Module):
    def __init__(self,num_classes):
        super(VGG16, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential (
            
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            )
        self.classifier = nn.Sequential(
        	nn.Linear(512*7*7,4096),
        	nn.ReLU(True),
        	nn.Dropout(),
        	nn.Linear(4096,4096),
        	nn.ReLU(True),
        	nn.Dropout(),
        	nn.Linear(4096,num_classes),
        	)

class BinVGG16 (nn.Module):
    def __init__(self,num_classes):
        super(BinVGG16, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential (
            
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),


            BinConv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            BinConv2d(64, 128, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            
            BinConv2d(128, 256, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1),
            nn.MaxPool2d(kernel_size=2,stride=2),

            BinConv2d(256,512, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(512,512, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(512,512, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(512,512, kernel_size=3, stride=1, padding=1, groups=1),
            nn.MaxPool2d(kernel_size=2,stride=2),

            BinConv2d(512,512, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(512,512, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(512,512, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(512,512, kernel_size=3, stride=1, padding=1, groups=1),
            nn.MaxPool2d(kernel_size=2,stride=2),

            )
        self.classifier = nn.Sequential(
            BinConv2d(512*7*7, 4096, Linear=True),
            BinConv2d(4096, 4096, dropout=0.5, Linear=True),
            nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True),
            nn.Dropout(),
            nn.Linear(4096,num_classes),
            )

class VGG16Detec (nn.Module):
    def __init__(self):
        super(VGG16Detec, self).__init__()
        self.features = nn.Sequential (
            
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1470),
            )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 512*7*7)
        #print ('feature size is ',x.size(1))
        x = self.classifier(x)
        x = F.sigmoid(x)
        x = x.view(-1,7,7,30)
        return x    


class BinVGG16Detec (nn.Module):
    def __init__(self):
        super(BinVGG16Detec, self).__init__()
        self.features = nn.Sequential (
            
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64,eps=1e-4, momentum=0.1, affine=True), 
            nn.ReLU(inplace=True),


            BinConv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            BinConv2d(64, 128, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            
            BinConv2d(128, 256, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=1),
            nn.MaxPool2d(kernel_size=2,stride=2),

            BinConv2d(256,512, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(512,512, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(512,512, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(512,512, kernel_size=3, stride=1, padding=1, groups=1),
            nn.MaxPool2d(kernel_size=2,stride=2),

            BinConv2d(512,512, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(512,512, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(512,512, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(512,512, kernel_size=3, stride=1, padding=1, groups=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.MaxPool2d(kernel_size=2,stride=2)

            )
        self.classifier = nn.Sequential(
            BinConv2d(512*7*7, 4096, Linear=True),
            nn.Linear(4096, 1470)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 512*7*7)
        #print ('feature size is ',x.size(1))
        x = self.classifier(x)
        x = F.sigmoid(x)
        x = x.view(-1,7,7,30)
        return x

def vgg(pretrained=False):
    model = BinVGG16Detec()
    if pretrained:
        model_path = 'model_list/alexnet.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
    return model

def vggDetecFP(pretrained=False):
    model = VGG16Detec()
    if pretrained:
        model_path = 'model_list/alexnet.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
    return model