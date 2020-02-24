import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    def __init__(self, n=2, num_classes=10, batch_size=1, no_avgpool=True, weights=None, stats=None, widths=None, segment=1, reduce_ip='127.0.0.1', reduce_port=None, id=0, id_stu=0):

        super(WRN_extract,self).__init__()
        self.id = id
        self.batch_size = batch_size
        self.segment = segment
        self.id_stu  = id_stu

        if segment > 1:
            self.bcast_port = create_bcast_port( reduce_ip, reduce_port )
        else:
            self.bcast_port = None

        self.conv0 = nn.Conv2d(3,int(16/segment),3, padding=1, bias=False)
        self.groups = []
        self.g0 = Group(16, int(widths[0]), n=n, segment=segment, id=id, bcast_port=self.bcast_port)
        self.g1 = Group( int(widths[0]), no=int(widths[1]), n=n, stride=2 )
        self.g2 = Group( int(widths[1]), no=int(widths[2]), n=n, stride=2 )
        self.conv_g2_dimComm = nn.Conv2d( int(widths[2]), int(widths[3]), 1, bias=False)
        postfix = 'a'
        if id_stu is 0:
            # postfix = 
            self.groups_info = [('group0',self.g0), ('group1a',self.g1), ('group2a',self.g2)]
        elif id_stu is 1:
            self.groups_info = [('group0',self.g0), ('group1b',self.g1), ('group2b',self.g2)]
        else:
            raise ValueError()

        if weights is not None:
            self.load_weight(weights, prefix='student.')

        if stats is not None:
            self.load_stats(stats, prefix='student.')

    def __del__(self):
        if hasattr( self, 'c' ):
            print("Close the reduce port")
            self.c.close()

    def load_weight( self, param, prefix="" ):
        assert( self.conv0.weight.shape == param[prefix+'conv0'].shape)
        self.conv0.weight = nn.Parameter(param[prefix+'conv0'])
        if self.id_stu is 0:
            assert( self.conv_g2_dimComm.weight.shape == param[prefix+'conv_g2a_dimComm'].shape)
            self.conv_g2_dimComm.weight = nn.Parameter(param[prefix+'conv_g2a_dimComm'])
        elif self.id_stu is 1:
            assert( self.conv_g2_dimComm.weight.shape == param[prefix+'conv_g2b_dimComm'].shape)
            self.conv_g2_dimComm.weight = nn.Parameter(param[prefix+'conv_g2b_dimComm'])
        for sub_prefix, g in self.groups_info:
            g.set_param( param, prefix+sub_prefix )

    def load_stats( self, stats, prefix="" ):
        for sub_prefix, g in self.groups_info:
            g.set_stats( stats, prefix+sub_prefix )

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
        # x = self.bcast_data(x_split)

        x_split = split(x_split,self.segment,self.id,dim=1)
        x_split = self.g0.forward(x_split)
        x_split = self.g1.forward(x_split)
        # x_r = self.g1b.forward(x_split)
        x_split = self.g2.forward(x_split)
        # x_r = self.g2b.forward(x_r)
        x_split = self.conv_g2_dimComm(x_split)
        # x_r = self.conv_g2b_dimComm(x_r)
        return x_split

    def set_requires_grad( self, val ):
        for g in [self.g0, self.g1, self.g2]:
            g.set_requires_grad(val)
        for para in self.parameters():
            para.requires_grad = val
        return

    def eval( self ):
        for g in [self.g0, self.g1, self.g2]:
            g.eval()
        return self.train(False)

class WRN_fc(nn.Module):

    def __init__(self, n=6, num_classes=10, batch_size=1, no_avgpool=True, weights=None, stats=None, widths=torch.Tensor([16,32,64]).mul(4)):
        super(WRN_fc,self).__init__()

        widths = torch.Tensor([32, 64, 64, 87, 87, 34, 46]).int()
        self.n = n
        self.batch_size = batch_size
        self.fc       = nn.Linear( 80, num_classes )
        self.bn       = nn.BatchNorm2d(int(widths[5]+widths[6]))
        self.relu     = nn.ReLU(inplace=True)

        if no_avgpool:
            self.avg_pool = AvgPool2d_in_conv(int(widths[5]+widths[6]),8)
        else:
            self.avg_pool = nn.AvgPool2d(8,1,0)



        if weights is not None:
            self.load_weight(weights, prefix='student.')

        if stats is not None:
            self.load_stats(stats, prefix='student.')

    def load_weight( self, param, prefix ):
        self.fc.weight = nn.Parameter(param[prefix+'fc.weight'])
        self.fc.bias   = nn.Parameter(param[prefix+'fc.bias'])
        assert( self.bn.weight.shape == param[prefix+'bn.weight'].shape)
        self.bn.weight = nn.Parameter(param[prefix+'bn.weight'])
        assert( self.bn.bias.shape == param[prefix+'bn.bias'].shape)
        self.bn.bias   = nn.Parameter(param[prefix+'bn.bias'])

    def load_stats( self, stats, prefix ):
        assert( self.bn.running_mean.shape == stats[prefix+'bn.running_mean'].shape)
        self.bn.running_mean = nn.Parameter(stats[prefix+'bn.running_mean'])
        assert( self.bn.running_var.shape == stats[prefix+'bn.running_var'].shape)
        self.bn.running_var = nn.Parameter(stats[prefix+'bn.running_var'])
        return

    def forward(self, x):
        x_split = self.bn(x)
        x_split = self.relu(x_split)
        x = self.avg_pool(x_split)
        x = self.fc(x.view(self.batch_size,80))
        return x

    def set_requires_grad( self, val ):
        for para in self.parameters():
            para.requires_grad = val

class WRN(nn.Module):

    def __init__(self, n=6, num_classes=10, batch_size=1, no_avgpool=True, weights=None, stats=None, widths=torch.Tensor([16,32,64]).mul(4)):
        super(WRN,self).__init__()
        self.extr0 = WRN_extract( weights=param['params'], stats=param['stats'], widths=torch.Tensor([32, 64, 87, 34]), id_stu=0)
        self.extr1 = WRN_extract( weights=param['params'], stats=param['stats'], widths=torch.Tensor([32, 64, 87, 46]), id_stu=1)
        self.fc    = WRN_fc(      weights=param['params'], stats=param['stats'] )

    def forward(self, x):
        x1 = self.extr0(x)
        x2 = self.extr1(x)
        x = torch.cat((x1,x2), dim=1)
        return self.fc(x)
        x_split = self.bn(x)
        x_split = self.relu(x_split)
        x = self.avg_pool(x_split)
        x = self.fc(x.view(self.batch_size,80))
        return x

    def eval( self ):
        for m in [self.extr0, self.extr1, self.fc]:
            m.eval()

    def set_requires_grad( self, val ):
        for m in [self.extr0, self.extr1, self.fc]:
            m.set_requires_grad(val)
        for para in self.parameters():
            para.requires_grad = val

import argparse

def get_dataset( dataset_name ):
    if dataset_name == "CIFAR10":
        import torchvision
        dataset = torchvision.datasets.CIFAR10( args.cifar, train=False, download=True)
    return dataset

def normalize_unsqueeze( img, mean, std ):
    import torch
    import torchvision
    to_tensor = torchvision.transforms.ToTensor()
    img_tensor = to_tensor(img).transpose(0,2)
    img_tensor = img_tensor-(torch.Tensor(mean)/256)
    img_tensor = img_tensor/(torch.Tensor( std)/256)
    img_tensor = img_tensor.transpose(0,2)
    return img_tensor.unsqueeze(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt7_path",   help="Path of *.pt7", type=str, required=True)
    parser.add_argument("--batch_size", help="Batch size",    type=int, default=1)
    parser.add_argument("--to_onnx",    help="Compile to onnx or not", action='store_true')
    parser.add_argument("--output",     help="Output name",   type=str, default="wrn")
    parser.add_argument("--cifar",      help="Path of CIFAR", type=str, default='.')
    parser.add_argument("--dataset",    help="Dataset",       type=str, default='CIFAR10')
    parser.add_argument("--inference",  help="", action='store_true')
    args = parser.parse_args()

    # Define the model
    # net = Net( no_avgpool= args.to_onnx, batch_size=args.batch_size )
    param = torch.load(  args.pt7_path, map_location='cpu' )
    extr0 = WRN_extract( batch_size=args.batch_size, weights=param['params'], stats=param['stats'], widths=torch.Tensor([32, 64, 87, 34]), id_stu=0)
    extr1 = WRN_extract( batch_size=args.batch_size, weights=param['params'], stats=param['stats'], widths=torch.Tensor([32, 64, 87, 46]), id_stu=1)
    fc    = WRN_fc(      batch_size=args.batch_size, weights=param['params'], stats=param['stats'] )
    model = WRN( weights=param['params'], stats=param['stats'] )

    # Load parameters    
    param = torch.load( args.pt7_path, map_location='cpu' )
    for m in [extr0, extr1, fc, model]:
        # m.load_weight(param['params'],)
        # m.load_stats( param['stats'])
        m.set_requires_grad(False)
        m.eval()

    if args.to_onnx:
        from torch.autograd import Variable
        import torch.onnx

        dummy_input = Variable(torch.randn(args.batch_size, 3, 32, 32))
        torch.onnx.export(extr0, dummy_input, args.output+"_g1.onnx", verbose=True)

        dummy_input = Variable(torch.randn(args.batch_size, 3, 32, 32))
        torch.onnx.export(extr1, dummy_input, args.output+"_g2.onnx", verbose=True)

        dummy_input = Variable(torch.randn(args.batch_size, 80, 8, 8))
        torch.onnx.export(fc, dummy_input, args.output+"_fc.onnx", verbose=True)

#        dummy_input = Variable(torch.randn(args.batch_size, 3, 32, 32))
#        torch.onnx.export(model, dummy_input, args.output+"_all.onnx", verbose=True)

    if args.inference:
        cifar = get_dataset( args.dataset )

        import queue
        q_target = queue.Queue(2)
        
        correct = 0
        total   = 0
        for batch_idx, (inputs, targets) in enumerate(cifar):
            normed    = normalize_unsqueeze(inputs, [125.3, 123.0, 113.9], [63, 62.1, 66.7])
            ftr0 = extr0(normed)
            ftr1 = extr1(normed)
            ftrs = torch.cat((ftr0,ftr1), dim=1)
            print("shape = ", ftr0.shape, ftr1.shape)
            scrs = fc(ftrs)
            predicted = np.argmax(scrs.detach().numpy())
            total   += 1
            correct += predicted==targets
            print(correct,'/',total)

                        # predicted = np.argmax(outputs)
