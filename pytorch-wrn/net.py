import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock( nn.Module ):
    def __init__( self, ni, no, stride=1, convdim=False ):
        super( BasicBlock, self ).__init__()
        self.bn0     = nn.BatchNorm2d(ni)
        self.relu0   = nn.ReLU(inplace=True)
        self.conv0   = nn.Conv2d( ni, no, 3, padding=1, stride=stride, bias=False )
        self.bn1     = nn.BatchNorm2d(no)
        self.relu1   = nn.ReLU(inplace=True)
        self.conv1   = nn.Conv2d( no, no, 3, stride=1, padding=1, bias=False )
        self.convdim = nn.Conv2d(ni,no,1,stride=stride, bias=False) if convdim else None

    def forward( self, x ):
        
        residual = x

        x = self.bn0(x)
        x = self.relu0(x)
        out = self.conv0(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv1(out)

        if self.convdim is not None:
            residual = self.convdim(x)

        out += residual

        return out

    def set_param( self, param, prefix ):
        if self.convdim is not None:
            self.convdim.weight = nn.Parameter(param[ prefix+'.convdim' ])
        self.bn0.weight   = nn.Parameter(param[ prefix+'.bn0.weight' ])
        self.bn0.bias     = nn.Parameter(param[ prefix+'.bn0.bias' ])
        self.conv0.weight = nn.Parameter(param[ prefix+'.conv0'])
        self.bn1.weight   = nn.Parameter(param[ prefix+'.bn1.weight' ])
        self.bn1.bias     = nn.Parameter(param[ prefix+'.bn1.bias' ])
        self.conv1.weight = nn.Parameter(param[ prefix+'.conv1'])

    def set_stats( self, stats, prefix ):
        self.bn0.running_mean = nn.Parameter(stats[ prefix+'.bn0.running_mean' ])
        self.bn0.running_var = nn.Parameter(stats[ prefix+'.bn0.running_var' ])
        self.bn1.running_mean = nn.Parameter(stats[ prefix+'.bn1.running_mean' ])
        self.bn1.running_var = nn.Parameter(stats[ prefix+'.bn1.running_var' ])

def Group( ni, no, n, stride=1 ):
    blocks = []
    blocks.append( BasicBlock(ni,no,stride,convdim=True) )
    for _ in range(1,n):
        blocks.append( BasicBlock(no,no) )
    return nn.Sequential(*blocks)


def AvgPool2d_in_conv( n, kernel_size ):
    avg = nn.Conv2d( n, n, kernel_size, bias=False)
    avg_weight = torch.zeros_like(avg.weight)
    for i in range(n):
        avg_weight[i][i] = torch.ones_like(avg.weight[0][0])
    avg.weight = nn.Parameter(avg_weight)
    return avg

class Net(nn.Module):

    def __init__(self, n=6, num_classes=10, batch_size=1, no_avgpool=True, weights=None, stats=None, widths=torch.Tensor([16,32,64]).mul(4)):
        super(Net,self).__init__()
        self.n = n
        self.batch_size = batch_size
        self.conv0 = nn.Conv2d(3,16,3, padding=1, bias=False)
        groups = [None] * 3
        groups[0] =  Group(             16, no=int(widths[0]), n=n )
        groups[1] =  Group( int(widths[0]), no=int(widths[1]), n=n, stride=2 )
        groups[2] =  Group( int(widths[1]), no=int(widths[2]), n=n, stride=2 )
        self.groups   = nn.Sequential(*groups)

        self.bn       = nn.BatchNorm2d(int(widths[2]))
        self.relu     = nn.ReLU(inplace=True)

        if no_avgpool:
            self.avg_pool = AvgPool2d_in_conv(int(widths[2]),8)
        else:
            self.avg_pool = nn.AvgPool2d(8,1,0)

        self.fc       = nn.Linear( int(widths[2]), num_classes )

        if weights is not None:
            self.load_weight(weights)

        if stats is not None:
            self.load_stats(stats)

    def load_weight( self, param ):
        self.conv0.weight = nn.Parameter(param['conv0'])

        for i in range(3):
            for j in range(self.n):
                prefix = 'group' + str(i) + '.block' + str(j)
                self.groups[i][j].set_param( param, prefix )

        self.bn.weight = nn.Parameter(param['bn.weight'])
        self.bn.bias   = nn.Parameter(param['bn.bias'])
        self.fc.weight = nn.Parameter(param['fc.weight'])
        self.fc.bias   = nn.Parameter(param['fc.bias'])

    def load_stats( self, stats ):

        for i in range(3):
            for j in range(self.n):
                prefix = 'group' + str(i) + '.block' + str(j)
                self.groups[i][j].set_stats( stats, prefix )

        self.bn.running_mean = nn.Parameter(stats['bn.running_mean'])
        self.bn.running_var = nn.Parameter(stats['bn.running_var'])

    def forward(self, x):
        x = self.conv0(x)
        x = self.groups[0](x)
        x = self.groups[1](x)
        x = self.groups[2](x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.fc(x.view(self.batch_size,256))
        return x

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt7_path",   help="Path of *.pt7", type=str, required=True)
    parser.add_argument("--batch_size", help="Batch size",    type=int, default=1)
    parser.add_argument("--to_onnx",    help="Compile to onnx or not", action='store_true')
    parser.add_argument("--output",     help="Output name",   type=str, default="wrn")
    args = parser.parse_args()

    # Define the model
    net = Net( no_avgpool= args.to_onnx, batch_size=args.batch_size )

    # Load parameters    
    param = torch.load( args.pt7_path, map_location='cpu' )
    net.load_weight(param['params'])
    net.load_stats( param['stats'])
    for param in net.parameters():
        param.requires_grad = False
    print(net)

    if args.to_onnx:
        from torch.autograd import Variable
        import torch.onnx
        dummy_input = Variable(torch.randn(args.batch_size, 3, 32, 32))
        torch.onnx.export(net, dummy_input, args.output+".onnx", verbose=True)
