import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch.nn.functional as F
import util

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
        self.relu = nn.LeakyReLU(inplace=True)
    
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

class VGG(nn.Module):

    def __init__(self, features_fixed_0,features_xnor,features_fixed_1, num_classes=1470, image_size=448):
        super(VGG, self).__init__()
        self.features_fixed_0 = features_fixed_0
        self.features_xnor = features_xnor
        self.features_fixed_1 = features_fixed_1
        self.image_size = image_size
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features_fixed_0(x)
        x = self.features_xnor(x)
        x = self.features_fixed_1(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.sigmoid(x) 
        x = x.view(-1,7,7,30)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    if str(cfg[0]) == '512B' or str(cfg[0]) == '512':
        in_channels = 512
    for v in cfg:
        
        if str(v)[-1] == 'B':
            cout = int(v[:-1])
            layers += [BinConv2d(in_channels,cout,kernel_size=3, stride=1, padding=1, groups=1)]
            in_channels = cout
        elif v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def conv_bn_relu(in_channels,out_channels,kernel_size=3,stride=2,padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'F': [64, '64B', 'M', '128B', '128B', 'M', '256B', '256B', '256B', 'M', '512B', '512B', '512B', 'M', '512B', '512B', 512, 'M'],

    '1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    '2': ['512B', '512B'],
    '3': [512, 'M']

}


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model

def vgg16XNOR(pretrained=False, **kwargs):
   
    model = VGG(make_layers(cfg['1']),make_layers(cfg['2']),make_layers(cfg['3']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model

def test():
    import torch
    from torch.autograd import Variable
    model = vgg16XNOR()
    print(model)
    bin_range= [1,11]
    bin_op = util.BinOp(model,bin_range)
    img = torch.rand(2,3,224,224)
    img = Variable(img)
    feature, output= model(img)
    print(output.size())
    print(feature.size())

if __name__ == '__main__':
    test()