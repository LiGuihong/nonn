import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as dsets
import torch.backends.cudnn as cudnn

from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import transforms, datasets
import random
import numpy as np
import argparse,os,PIL
import datetime
#from __future__ import print_function, absolute_import


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--exp_groups', default=0, type=int,
                    help='experiments groups')
parser.add_argument('--tc_groups', default=0, type=int,
                            help='experiments groups')

parser.add_argument('--image_size', default=224, type=int,
                    help='image size')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='use advprop or not')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--wm', default=3.0, type=float,
                            help='width_multiplier')
args = parser.parse_args()
lr=args.lr
momentum=args.momentum
weight_decay=args.weight_decay
batch_size=args.batch_size
workers=args.workers
print_freq=args.print_freq

class conv_depth(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=32,kernel_size=[3,1], stride=[1,1], padding=[1,0],bias=[False,False]):
        super(conv_depth, self).__init__()
        self.conv0=nn.Conv2d(in_channels=in_channels, out_channels=int(out_channels*args.wm), kernel_size=1, 
                            stride=1, padding=0,groups=1,bias=False)
        self.batch0=nn.BatchNorm2d(int(out_channels*args.wm))
        self.conv1=nn.Conv2d(in_channels=int(out_channels*args.wm), out_channels=int(out_channels*args.wm), kernel_size=kernel_size[0], 
                            stride=stride[0], padding=padding[0],groups=int(out_channels*args.wm),bias=bias[0])

        self.batch1=nn.BatchNorm2d(int(out_channels*args.wm))
        self.conv2=nn.Conv2d(in_channels=int(out_channels*args.wm), out_channels=out_channels, kernel_size=kernel_size[1], 
                    stride=stride[1], padding=padding[1],groups=1,bias=bias[1])        
        self.batch2=nn.BatchNorm2d(out_channels)
        self.stride=stride[0]
        #self.conv3=nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size[0], 
        #                    stride=stride[0], padding=padding[0],groups=1,bias=False)
        #self.batch3=nn.BatchNorm2d(out_channels)
        self.out_channels=out_channels
    def forward(self, x):
        out=F.relu(self.batch0(self.conv0(x)))
        out=F.relu(self.batch1(self.conv1(out)))
        out=self.batch2(self.conv2(out))
        if self.stride!=2:
            out=out+x[:,0:self.out_channels,:,:]
        #out=self.batch3(self.conv3(x))+out
        # out=64.0*out
        return out 

class SimpleNet_order_no_batch(torch.nn.Module):
    def __init__(self,cell_depth=10,width_mul=2,tc_array=[10,35,50,70],num_classes=10,num_cells=4):
        super(SimpleNet_order_no_batch, self).__init__()
        self.width_base=np.array(np.array([32,64,128,256]),dtype=int)
        self.all_path_num=np.zeros([num_cells,cell_depth])
        self.layer_tc=np.zeros([num_cells,cell_depth])
        self.nn_mass=0
        self.density=np.zeros(num_cells)
        self.cell_depth=cell_depth
        for k in range(num_cells):
            for i in range(cell_depth-2):
                for j in range(i+1):
                    self.all_path_num[k,i+2]=self.all_path_num[k,i+2]+self.width_base[k]
                self.layer_tc[k,i+2]=min(tc_array[k],self.all_path_num[k,i+2])
            self.layer_tc=np.array(self.layer_tc,dtype=int)
            self.all_path_num=np.array(self.all_path_num,dtype=int)
            self.density[k]=(np.sum(self.layer_tc[k]))/(np.sum(self.all_path_num[k]))
            self.nn_mass=self.nn_mass+self.density[k]*self.width_base[k]*cell_depth

        self.layer_tc=np.array(self.layer_tc,dtype=int)
        self.all_path_num=np.array(self.all_path_num,dtype=int)

        cell_idx=0
        layer_list1=[]
        layer_list1.append(nn.Conv2d(in_channels=3, out_channels=self.width_base[cell_idx], 
                                    kernel_size=7, stride=2, padding=3,bias=False))
        
        for i in range(cell_depth-1):
            layer_list1.append(conv_depth(in_channels=self.width_base[cell_idx]+self.layer_tc[cell_idx,i+1], 
                                kernel_size=[3,1],out_channels=self.width_base[cell_idx]))
        # layer_list1.append(nn.Linear(net_arch[net_depth-1]+layer_tc[net_depth-1], 2))
        self.features1 = nn.ModuleList(layer_list1).eval() 
        self.link_dict1=[]
        for i in range(cell_depth):
            self.link_dict1.append(self.add_link(cell_idx=cell_idx,idx=i))


        cell_idx=1
        layer_list2=[]
        layer_list2.append(conv_depth(in_channels=self.width_base[cell_idx-1], out_channels=self.width_base[cell_idx],stride=[2,1]))

        for i in range(cell_depth-1):
            layer_list2.append(conv_depth(in_channels=self.width_base[cell_idx]+self.layer_tc[cell_idx,i+1], 
                                out_channels=self.width_base[cell_idx]))
        # layer_list2.append(nn.Linear(net_arch[net_depth-1]+layer_tc[net_depth-1], 2))
        self.features2 = nn.ModuleList(layer_list2).eval() 
        self.link_dict2=[]
        for i in range(cell_depth):
            self.link_dict2.append(self.add_link(cell_idx=cell_idx,idx=i))

        cell_idx=2
        layer_list3=[]
        layer_list3.append(conv_depth(in_channels=self.width_base[cell_idx-1], 
                            out_channels=self.width_base[cell_idx],stride=[2,1]))

        for i in range(cell_depth-1):
            layer_list3.append(conv_depth(in_channels=self.width_base[cell_idx]+self.layer_tc[cell_idx,i+1], 
                                out_channels=self.width_base[cell_idx]))

        # layer_list3.append(nn.Linear(net_arch[net_depth-1]+layer_tc[net_depth-1], 2))
        self.features3 = nn.ModuleList(layer_list3).eval() 
        self.link_dict3=[]
        for i in range(cell_depth):
            self.link_dict3.append(self.add_link(cell_idx=cell_idx,idx=i))


        cell_idx=3
        layer_list4=[]
        layer_list4.append(conv_depth(in_channels=self.width_base[cell_idx-1], out_channels=self.width_base[cell_idx],stride=[2,1]))
        for i in range(cell_depth-1):
            layer_list4.append(conv_depth(in_channels=self.width_base[cell_idx]+self.layer_tc[cell_idx,i+1], 
                                out_channels=self.width_base[cell_idx]))
        # layer_list4.append(nn.Linear(net_arch[net_depth-1]+layer_tc[net_depth-1], 2))
        self.features4 = nn.ModuleList(layer_list4).eval() 
        self.link_dict4=[]
        for i in range(cell_depth):
            self.link_dict4.append(self.add_link(cell_idx=cell_idx,idx=i))


        self.avg_pool=nn.AvgPool2d(kernel_size=7,stride=1,  padding=0)
        self.fc1=nn.Linear(in_features=self.width_base[3], out_features=num_classes, bias=True)
        self._initialize_weights()


    def add_link(self,cell_idx,idx=0):
        tmp=list((np.arange(self.all_path_num[cell_idx,idx])))
        random.seed(2)
        link_idx=random.sample(tmp,self.layer_tc[cell_idx,idx])
        link_params=torch.tensor(link_idx)
        return link_params

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        cell_depth=self.cell_depth
        cell_idx=0
        out0=self.features1[0](x)
        out0=F.max_pool2d(out0,  kernel_size=3, stride=2)
        out_dict=out0
        out1=F.relu(out0)       
        out1=self.features1[1](out1)
        out_dict=out1
        feat_dict=[]
        feat_dict=(out0)
        # print(out0.size(),out1.size())
        feat_dict=(torch.cat((out1,out0),1))
        for layer_idx in range(cell_depth-2):
            in_features=feat_dict
            if self.layer_tc[cell_idx, layer_idx+2]>0:
                in_tmp=torch.cat((out_dict,in_features[:,self.link_dict1[layer_idx+2],:,:]),1)
                if layer_idx<cell_depth-2:
                    out_tmp=self.features1[layer_idx+2](in_tmp)
                    feat_dict=(torch.cat((out_tmp,feat_dict),1))
                    out_dict=out_tmp
                else:
                    out_tmp=self.features1[layer_idx+2](in_tmp)
                    feat_dict=(torch.cat((out_tmp,feat_dict),1))
                    out_dict=out_tmp           
            else:
                in_tmp=out_dict
                if layer_idx<cell_depth-2:
                    out_tmp=self.features1[layer_idx+2](in_tmp)
                    feat_dict=(torch.cat((out_tmp,feat_dict),1))
                    out_dict=out_tmp
                else:
                    out_tmp=self.features1[layer_idx+2](in_tmp)
                    feat_dict=(torch.cat((out_tmp,feat_dict),1))
                    out_dict=out_tmp   

        x2=out_dict
        # print(x2.size())
        cell_idx=1
        out0=self.features2[0](x2)
        out_dict=out0

        out1=self.features2[1](out0)
        out_dict=out1
        feat_dict=[]
        feat_dict=out0
        feat_dict=(torch.cat((out1,out0),1))
        for layer_idx in range(cell_depth-2):
            in_features=feat_dict[layer_idx]
            if self.layer_tc[cell_idx, layer_idx+2]>0:
                in_tmp=torch.cat((out_dict,in_features[:,self.link_dict2[layer_idx+2],:,:]),1)
                if layer_idx<cell_depth-2:
                    out_tmp=self.features2[layer_idx+2](in_tmp)
                    feat_dict=(torch.cat((out_tmp,feat_dict),1))
                    out_dict=out_tmp
                else:
                    out_tmp=self.features2[layer_idx+2](in_tmp)
                    feat_dict=(torch.cat((out_tmp,feat_dict),1))
                    out_dict=out_tmp           
            else:
                in_tmp=out_dict
                if layer_idx<cell_depth-2:
                    out_tmp=self.features2[layer_idx+2](in_tmp)
                    feat_dict=(torch.cat((out_tmp,feat_dict),1))
                    out_dict=out_tmp
                else:
                    out_tmp=self.features2[layer_idx+2](in_tmp)
                    feat_dict=(torch.cat((out_tmp,feat_dict),1))
                    out_dict=out_tmp   

        x3=out_dict
        cell_idx=2
        out0=self.features3[0](x3)
        out_dict=out0
  
        out1=self.features3[1](out0)
        out_dict=out1
        feat_dict=[]
        feat_dict=out0
        feat_dict=torch.cat((out1,out0),1)
        for layer_idx in range(cell_depth-2):
            in_features=feat_dict[layer_idx]
            if self.layer_tc[cell_idx, layer_idx+2]>0:
                in_tmp=torch.cat((out_dict,in_features[:,self.link_dict3[layer_idx+2],:,:]),1)
                if layer_idx<cell_depth-2:
                    out_tmp=self.features3[layer_idx+2](in_tmp)
                    feat_dict=(torch.cat((out_tmp,feat_dict),1))
                    out_dict=out_tmp
                else:
                    out_tmp=self.features3[layer_idx+2](in_tmp)
                    feat_dict=(torch.cat((out_tmp,feat_dict),1))
                    out_dict=out_tmp           
            else:
                in_tmp=out_dict
                if layer_idx<cell_depth-2:
                    out_tmp=self.features3[layer_idx+2](in_tmp)
                    feat_dict=(torch.cat((out_tmp,feat_dict),1))
                    out_dict=out_tmp
                else:
                    out_tmp=self.features3[layer_idx+2](in_tmp)
                    feat_dict=(torch.cat((out_tmp,feat_dict),1))
                    out_dict=out_tmp  


        x4=out_dict[cell_depth-1]
        cell_idx=3
        out0=self.features4[0](x4)
        out_dict=out0

        out1=self.features4[1](out0)
        out_dict=out1
        feat_dict=[]
        feat_dict=out0
        feat_dict=(torch.cat((out1,out0),1))
        for layer_idx in range(cell_depth-2):
            in_features=feat_dict[layer_idx]
            if self.layer_tc[cell_idx, layer_idx+2]>0:
                in_tmp=torch.cat((out_dict,in_features[:,self.link_dict4[layer_idx+2],:,:]),1)
                if layer_idx<cell_depth-2:
                    out_tmp=self.features4[layer_idx+2](in_tmp)
                    feat_dict=torch.cat((out_tmp,feat_dict),1)
                    out_dict=out_tmp
                else:
                    out_tmp=self.features4[layer_idx+2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict=out_tmp           
            else:
                in_tmp=out_dict[layer_idx+1]
                if layer_idx<cell_depth-2:
                    out_tmp=self.features4[layer_idx+2](in_tmp)
                    feat_dict.append(torch.cat((out_tmp,feat_dict[layer_idx+1]),1))
                    out_dict=out_tmp
                else:
                    out_tmp=self.features4[layer_idx+2](in_tmp)
                    feat_dict=(torch.cat((out_tmp,feat_dict),1))
                    out_dict=out_tmp  



        out=self.avg_pool(out_dict)
        batch_size=out.size()[0]
        out=self.fc1(out.view(batch_size,-1))
        return out

# net=SimpleNet_order_no_batch(cell_depth=20,width_mul=2,tc_array=[10,10,10,10],num_classes=10,num_cells=4)
# data=torch.zeros([10,3,224,224])
# print(net(data))
# exit()

if (torch.cuda.is_available()):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
criterion = nn.CrossEntropyLoss()
criterion_sum = nn.CrossEntropyLoss(reduction='sum')

def accuracy(output, target, topk=(1,5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    new_output=output.detach().cpu()
    new_target=target.detach().cpu()
    batch_size = new_target.size(0)

    _, pred = new_output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(new_target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def train(images, labels, optimizer, model):
    images = Variable(images).to(device)
    labels = Variable(labels).to(device)

    optimizer.zero_grad()
    output = model(images)
    #print(outputs.size())
    #print(labels.size())
#    output = output.view((-1, 8) + output.size()[1:])
#    output = torch.mean(output, dim=1)
    # print(output.size(),labels.size())
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    # print(loss)
    return loss,output

test_num=100000
def test(epoch,test_loader,model,file_name):

    correct = 0.0
    total = 0.0
    total_loss = 0.0
    i=0
    predicted_label=torch.zeros(test_num)
    top1_acc=0
    top5_acc=0
    model.eval()
    for j,(images, labels) in enumerate(test_loader):        
        with torch.no_grad():
            #print(j)
            images = Variable(images).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            total += labels.size(0)
            top1,top5=accuracy(output=outputs, target=labels, topk=(1,5))
            #temp=(predicted .detach().cpu().numpy()== labels.detach().cpu().numpy()).sum()
            #correct=correct+temp
            total_loss=total_loss+loss.item()
            i=i+1
            top1_acc=top1_acc+top1
            top5_acc=top5_acc+top5
    print('epoch:{}  top1_acc:{:.4f}  top5_acc:{:.4f}  loss:{:.4f}'.format(epoch,top1_acc.item()/i,top5_acc.item()/i,total_loss/i))
  
    with open(file_name,'a+') as logs:
        #print(top1_acc/float(i),top5_acc/float(i), total_loss/float(i),file=logs)
        print('epoch:{}  top1_acc:{:.4f}  top5_acc:{:.4f}  loss:{:.4f}'.format(epoch,top1_acc.item()/i,top5_acc.item()/i,total_loss/i),file=logs)
    return 0,0

# tc_array=[[50,70,110,145],[10,20,30,50],[20,30,50,80],[30,50,80,100],[40,60,90,110],
#     [60,90,130,150],[20,40,50,70],[30,50,80,100],[40,70,90,110],[50,80,120,140],
#     [65,85,125,155],[15,35,55,80],[30,60, 80,100],[40,65,85,105],[45,75,115,135],[90,150,210],
#     [30,80,117],[50,110,150],[70,140,200],[90,175,250],[110,215,300]
# ]
# depth_array=[11,14,17]
# #nnmass =200,400,600,800
# for cell_depth in [0,1,2]:
#     print('===============')
#     base_idx=5*cell_depth
#     file_name='imgnet_'+str(cell_depth)+'_.logs'
#     for tc_group in range(5):
#         model = SimpleNet_order_no_batch(cell_depth=depth_array[cell_depth],width_mul=3,tc_array=tc_array[base_idx+tc_group],num_classes=1000,num_cells=4)
#         print(model.nn_mass)
    
# exit(0)

img_path='/home/ubuntu/imagenet/data/ILSVRC/Data/CLS-LOC/'
traindir = os.path.join(img_path, 'train')
# valdir = os.path.join(img_path, 'val')
testdir = os.path.join(img_path, 'val')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

image_size = 224
train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

test_dataset = datasets.ImageFolder(
    testdir,
    transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=workers)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=workers)


#tc_array=[[50,70,110,145],[10,20,30,50],[20,30,50,80],[30,50,80,100],[40,60,90,110],
#    [60,90,130,150],[20,40,50,70],[30,50,80,100],[40,70,90,110],[50,80,120,140],
#    [65,85,125,155],[15,35,55,80],[30,60, 80,100],[40,65,85,105],[45,75,115,135],[90,150,210],
#]
#tc_array=[[10,20,30,40]]

depth_array=[16]
tc_array=[[50,100,150,180]]
wm=3

for cell_depth in [args.exp_groups]:
    print('===================')
    base_idx=5*cell_depth
    file_name='imgnet_wm3_0_layers'+str(cell_depth)+'nnmass__.logs'
    test_name='imgnet_test_wm3_0_layers'+str(cell_depth)+'nnmass__.logs'
    for tc_group in [args.tc_groups]:
        model = SimpleNet_order_no_batch(cell_depth=depth_array[cell_depth],width_mul=wm,tc_array=tc_array[tc_group],num_classes=1000,num_cells=4)
        print(model.nn_mass,model.width_base*args.wm,depth_array[cell_depth])
        print('number of parameters: {}'.format( sum(p.numel() for p in model.parameters() if p.requires_grad)))

        with open(test_name,'a+') as logs:
            print(model.density,model.nn_mass,file=logs)
            print(cell_depth,tc_array[tc_group],file=logs)

        #if device == 'cuda':
        model = torch.nn.DataParallel(model)  
        cudnn.benchmark = True  
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        print(datetime.datetime.now())
        max_acc=-1000
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, float(args.epochs), eta_min=0.0)
        for epoch in range(args.epochs):
            print('epoch:{}'.format(epoch))
            correct = 0.0
            total = 0.0
            total_loss=0.0  
            i=0
            #test(epoch,test_loader,model,test_name)
            #if epoch %30==0 and epoch!=0:
            #    lr=lr*0.1
            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #                                optimizer, float(args.epochs), eta_min=0.0)
            #scheduler.step()
            #lr = scheduler.get_lr()[0]
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            temp_top1=0
            temp_top5=0
            temp_loss=0
            top1_acc=0
            top5_acc=0
            for j,(images, labels) in enumerate(train_loader):
                model.train()
                loss,outputs = train(images, labels,optimizer=optimizer,model=model)
                _, predicted = torch.max(outputs.data, 1)
                temp_loss=temp_loss+loss

                top1,top5=accuracy(output=outputs, target=labels, topk=(1,5))
                total += batch_size

                top1_acc=top1_acc+top1
                top5_acc=top5_acc+top5
                temp_top1=temp_top1+top1
                temp_top5=temp_top5+top5
                if j%print_freq==0 and j!=0: 
                    print('epoch:{}  step:{}  top1_acc:{:.4f}  top5_acc:{:.4f}  loss:{:.4f}'.format(epoch,j,temp_top1.item()/print_freq,temp_top5.item()/print_freq,temp_loss.item()/print_freq))
                    with open(file_name,'a+') as logs:
                        print('epoch:{}  step:{}  top1_acc:{:.4f}  top5_acc:{:.4f}  loss:{:.4f}'.format(epoch,j,temp_top1.item()/print_freq,temp_top5.item()/print_freq,temp_loss.item()/print_freq),file=logs)
                    print(datetime.datetime.now())
                    temp_top1=0
                    temp_top5=0
                    temp_loss=0
                total_loss=total_loss+loss
                i=i+1
            total_loss=total_loss/float(i)
            with open(file_name,'a+') as logs:
                #print(top1_acc/float(i),top5_acc/float(i),total_loss,file=logs)
                print('epoch:{}   top1_acc:{:.4f}  top5_acc:{:.4f}  loss:{:.4f}'.format(epoch,top1_acc.item()/i,top5_acc.item()/i,total_loss.item()/i),file=logs)
            test(epoch,test_loader,model,test_name)
            scheduler.step()
            lr = scheduler.get_lr()[0]
        PATH='ckpt/num_cell{}_cell_depth{}_wm_{}_tc_{}_{}_{}_'.format(4,depth_array[cell_depth],1.5,tc_array[tc_group][0],tc_array[tc_group][1],tc_array[tc_group][2],tc_array[tc_group][3])
        torch.save(model, PATH)
#'''


