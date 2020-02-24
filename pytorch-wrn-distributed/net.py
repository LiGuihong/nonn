import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlockUnit( nn.Module ):
    def __init__( self, ni, no, stride=1, convdim=False, segment=1, i=None, rd_buf=None, bcast_port=None ):
        super( BasicBlockUnit, self ).__init__()
        self.segment = segment
        self.ni      = ni
        self.no      = no
        self.id    = i
        self.bn0   = nn.BatchNorm2d(int(ni/self.segment))
        self.relu0 = nn.ReLU(inplace=True)
        self.conv0 = nn.Conv2d( ni, int(no/self.segment), 3, padding=1, stride=stride, bias=False )
        self.bn1   = nn.BatchNorm2d(int(no/self.segment))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d( no, int(no/self.segment), 3, stride=1, padding=1, bias=False )
        self.convdim = nn.Conv2d(ni,int(no/self.segment),1,stride=stride, bias=False) if convdim else None
        self.rd_buf = rd_buf
        self.bcast_port = bcast_port

    def bcast_data( self, x ):
        if self.segment is 1:
            return x
        import nn_format
        nn_format.send_a_tensor(self.bcast_port, x)
        x_merge_shape = torch.cat( tuple([x]*self.segment), 1 ).shape
        x_merge = nn_format.recv_as_tensor(self.bcast_port, x_merge_shape)
        assert(split(x_merge,self.segment,self.id,dim=1).equal(x))
        return x_merge

    def forward( self, x_split ):
        res_split = x_split
        x_split = self.bn0  (x_split)
        x_split = self.relu0(x_split)

        x = self.bcast_data(x_split)

        x1_split = self.conv0(x)
        x1_split = self.bn1  (x1_split)
        x1_split = self.relu1(x1_split)

        x1 = self.bcast_data(x1_split)

        x2_split = self.conv1(x1)
        if self.convdim is not None:
            res_split = self.convdim(x)
        x_split = x2_split + res_split

        return x_split

    def set_param( self, param, prefix ):
        i = self.id
        if self.convdim is not None:
            self.convdim.weight = nn.Parameter(split(param[ prefix+'.convdim' ], self.segment, self.id))
        self.bn0.weight   = nn.Parameter(split(param[ prefix+'.bn0.weight' ], self.segment, self.id))
        self.bn0.bias     = nn.Parameter(split(param[ prefix+'.bn0.bias' ], self.segment, self.id))
        self.conv0.weight = nn.Parameter(split(param[ prefix+'.conv0'], self.segment, self.id))
        self.bn1.weight   = nn.Parameter(split(param[ prefix+'.bn1.weight' ], self.segment, self.id))
        self.bn1.bias     = nn.Parameter(split(param[ prefix+'.bn1.bias' ], self.segment, self.id))
        self.conv1.weight = nn.Parameter(split(param[ prefix+'.conv1'], self.segment, self.id))

    def set_stats( self, stats, prefix ):
        i = self.id
        self.bn0.running_mean = nn.Parameter(split(stats[ prefix+'.bn0.running_mean' ], self.segment, self.id))
        self.bn0.running_var = nn.Parameter(split(stats[ prefix+'.bn0.running_var' ], self.segment, self.id))
        self.bn1.running_mean = nn.Parameter(split(stats[ prefix+'.bn1.running_mean' ], self.segment, self.id))
        self.bn1.running_var = nn.Parameter(split(stats[ prefix+'.bn1.running_var' ], self.segment, self.id))

    def set_requires_grad( self, val ):
        for para in self.parameters():
            para.requires_grad = val


def AvgPool2d_in_conv( n, kernel_size ):
    avg = nn.Conv2d( n, n, kernel_size, bias=False)
    avg_weight = torch.zeros_like(avg.weight)
    for i in range(n):
        avg_weight[i][i] = torch.ones_like(avg.weight[0][0])
    avg.weight = nn.Parameter(avg_weight)
    return avg

def split( tensor, segment, i, dim=0 ):
    if segment is 1:
        return tensor
    width = tensor.shape[dim]
    chunk = int(width/segment)
    if dim is 0:
        return tensor[i*chunk:(i+1)*chunk]
    elif dim is 1:
        return tensor[:,i*chunk:(i+1)*chunk]
    else:
        raise ValueError("Have not implemented yet.")


class Group(nn.Module):
    def __init__( self, ni, no, n, stride=1, segment=1, id=None, bcast_port=None ):
        super( Group, self ).__init__()
        self.blocks = []
        self.blocks.append( BasicBlockUnit(ni,no,stride,convdim=True,segment=segment,i=id,bcast_port=bcast_port) )
        for _ in range(1,n):
            self.blocks.append( BasicBlockUnit(no,no,segment=segment,i=id,bcast_port=bcast_port) )
    def forward( self, x ):
        for b in self.blocks:
            x = b.forward(x)
        return x
    def set_param( self, param, prefix ):
        for i, b in enumerate(self.blocks):
            b.set_param( param, prefix + '.block' + str(i) )
    def set_stats( self, stats, prefix ):
        for i, b in enumerate(self.blocks):
            b.set_stats( stats, prefix + '.block' + str(i) )

    def eval( self ):
        for b in self.blocks:
            b.eval()
        return self.train(False)

    def set_requires_grad( self, val ):
        for b in self.blocks:
            b.set_requires_grad(val)

def create_bcast_port( ip, port ):
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((ip, port))
    print("[+] Listening on {0}:{1}".format(ip, port))
    sock.listen(5)
    conn, addr = sock.accept()
    return conn

class WRN_extract(nn.Module):
    def __init__(self, n=6, num_classes=10, batch_size=1, no_avgpool=True, weights=None, stats=None, widths=torch.Tensor([16,32,64]).mul(4), segment=1, reduce_ip='127.0.0.1', reduce_port=None, id=0):

        super(WRN_extract,self).__init__()
        self.id = id
        self.batch_size = batch_size
        self.segment = segment

        self.bcast_port = create_bcast_port( reduce_ip, reduce_port )

        self.conv0 = nn.Conv2d(3,int(16/segment),3, padding=1, bias=False)
        self.groups = []
        self.groups.append( Group(16, int(widths[0]), 6, segment=segment, id=id, bcast_port=self.bcast_port))
        self.groups.append( Group( int(widths[0]), no=int(widths[1]), n=n, stride=2, id=id, bcast_port=self.bcast_port, segment=segment ) )
        self.groups.append( Group( int(widths[1]), no=int(widths[2]), n=n, stride=2, id=id, bcast_port=self.bcast_port, segment=segment ) )

        if weights is not None:
            self.load_weight(weights)

        if stats is not None:
            self.load_stats(stats)

    def __del__(self):
        if hasattr( self, 'c' ):
            print("Close the reduce port")
            self.c.close()

    def load_weight( self, param ):
        self.conv0.weight = nn.Parameter(param['conv0'][int(16/self.segment)*self.id:int(16/self.segment)*(self.id+1)])
        for i, g in enumerate(self.groups):
            prefix = 'group' + str(i)
            g.set_param( param, prefix )

    def load_stats( self, stats ):
        for i, g in enumerate(self.groups):
            prefix = 'group' + str(i)
            g.set_stats( stats, prefix )

    def bcast_data( self, x ):
        if self.segment is 1:
            return x
        import nn_format
        nn_format.send_a_tensor(self.bcast_port, x)
        x_merge_shape = torch.cat( tuple([x]*self.segment), 1 ).shape
        x_merge = nn_format.recv_as_tensor(self.bcast_port, x_merge_shape)
        assert(split(x_merge,self.segment,self.id,dim=1).equal(x))
        return x_merge

    def forward( self, x ):

        # x_split = split(x,self.segment,self.id,dim=1)

        x_split = self.conv0(x)
        x = self.bcast_data(x_split)

        x_split = split(x,self.segment,self.id,dim=1)
        for g in self.groups:
            x_split = g.forward(x_split)
        return x_split

    def set_requires_grad( self, val ):
        for g in self.groups:
            g.set_requires_grad(val)
        return

    def eval( self ):
        for g in self.groups:
            g.eval()
        return self.train(False)

class WRN_fc(nn.Module):

    def __init__(self, n=6, num_classes=10, batch_size=1, no_avgpool=True, weights=None, stats=None, widths=torch.Tensor([16,32,64]).mul(4)):
        super(WRN_fc,self).__init__()
        self.n = n
        self.batch_size = batch_size
        self.conv0 = nn.Conv2d(3,16,3, padding=1, bias=False)
        self.groups = []
        self.groups.append( Group(             16, no=int(widths[0]), n=n ) )
        self.groups.append( Group( int(widths[0]), no=int(widths[1]), n=n, stride=2 ) )
        self.groups.append( Group( int(widths[1]), no=int(widths[2]), n=n, stride=2 ) )

        # self.groups   = nn.Sequential(*groups)

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
        self.fc.weight = nn.Parameter(param['fc.weight'])
        self.fc.bias   = nn.Parameter(param['fc.bias'])
        for i, g in enumerate(self.groups):
            prefix = 'group' + str(i)
            g.set_param( param, prefix )
        self.bn.weight = nn.Parameter(param['bn.weight'])
        self.bn.bias   = nn.Parameter(param['bn.bias'])

    def load_stats( self, stats ):
        for i, g in enumerate(self.groups):
            prefix = 'group' + str(i)
            g.set_stats( stats, prefix )
        self.bn.running_mean = nn.Parameter(stats['bn.running_mean'])
        self.bn.running_var = nn.Parameter(stats['bn.running_var'])
        return

    def forward(self, x):
        # print("g0.x[0][0][0][0] = ", x[0][0][0][0])
        # x = self.groups[1].forward(x)
        print("g1.x[0][0][0][0] = ", x[0][0][0][0])
        # x = self.groups[2].forward(x)
        print("g2.x[0][0][0][0] = ", x[0][0][0][0])
        x = self.bn(x)
        print("bn.x[0][0][0][0] = ", x[0][0][0][0])
        x = self.relu(x)
        print("relu.x[0][0][0][0] = ", x[0][0][0][0])
        x = self.avg_pool(x)
        print("avg.x[0][0][0][0] = ", x[0][0][0][0])
        x = self.fc(x.view(self.batch_size,256))

        return x

    def set_requires_grad( self, val ):
        for para in self.parameters():
            para.requires_grad = val
        for g in self.groups:
            g.set_requires_grad(val)

    def eval( self ):
        for g in self.groups:
            g.eval()
        return self.train(False)

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