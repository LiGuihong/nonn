#!/usr/bin/env python
# coding: utf-8

def normalize_unsqueeze( img, mean, std ):
    import torch
    import torchvision
    to_tensor = torchvision.transforms.ToTensor()
    img_tensor = to_tensor(img).transpose(0,2)
    img_tensor = img_tensor-(torch.Tensor(mean)/256)
    img_tensor = img_tensor/(torch.Tensor( std)/256)
    img_tensor = img_tensor.transpose(0,2)
    return img_tensor.unsqueeze(0)

import nn_format
def client_send( data, sock ):

    import torch
    if type(data) is type(torch.Tensor()):
        flat_data = data.contiguous().view((-1,)).tolist()
    else:
        flat_data = data
    byte_flat_data = nn_format.list_to_bytestr(flat_data)
    byte_flat_data = nn_format.header_asbyte(byte_flat_data) + byte_flat_data
    sock.sendall(byte_flat_data)

def client_recv( sock ):
    response = nn_format.recv(sock)
    data_list = nn_format.bytestr_to_list(response)
    return data_list

import atexit
def exit_handler():
    for i, s in enumerate(socks):
        print("Close socket #%d(%s)" % (i,s))
        socks[s].close()

socks = {}

def get_dataset( dataset_name ):
    if dataset_name == "CIFAR10":
        import torchvision
        dataset = torchvision.datasets.CIFAR10( args.cifar, train=False, download=True)
    return dataset

def connects( socks, servers ):
    for dev in servers:
        try:
            ip = servers[dev]['ip']
            port = int(servers[dev]['port'])
            socks[dev] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            socks[dev].connect( (ip,port) )
        except ConnectionRefusedError:
            print( dev + " not found")


if __name__ == "__main__":

    atexit.register(exit_handler)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="Batch size",    type=int, default=1)
    parser.add_argument("--cifar",      help="Path of CIFAR", type=str, default='.')
    parser.add_argument("--config",     help="Configuration", type=str, default='config.xml')
    parser.add_argument("--dataset",    help="Dataset",       type=str, default='CIFAR10')
    parser.add_argument("--time_meas",  help="Measure time",  action='store_true')
    args = parser.parse_args()

    import socket
    import xmltodict
    with open( args.config ) as fd:
        doc = xmltodict.parse(fd.read())
        connects( socks, doc['config'] )

    import time
    correct = 0
    total = 0
    import numpy as np
    cifar = get_dataset( args.dataset )

    import queue
    q_target = queue.Queue(2)
    for batch_idx, (inputs, targets) in enumerate(cifar):
        normed    = normalize_unsqueeze(inputs, [125.3, 123.0, 113.9], [63, 62.1, 66.7])

        if args.time_meas:
            start_time = time.time()

        # if batch_idx >= 0:
        #     client_send( normed, socks['conv0'] )
        #     client_send( normed, socks['conv1'] )
        if batch_idx >= 0:
            client_send( normed,     socks['stage2L']   )
            client_send( normed,     socks['stage2R']   )
        if batch_idx >= 1:
            client_send( x2,     socks['fc']   )

        import torch
        # if batch_idx >= 0:
        #     x1_left = client_recv( socks['conv0'] )
        #     x1_rigt = client_recv( socks['conv1'] )

        #     x1_left_tensor = torch.Tensor(x1_left).view(1,8,32,32)
        #     x1_rigt_tensor = torch.Tensor(x1_rigt).view(1,8,32,32)
        #     x1 = torch.cat( tuple([x1_left_tensor,x1_rigt_tensor]), 1 )
        if batch_idx >= 0:
            x2_left = client_recv( socks['stage2L']   )
            x2_rigt = client_recv( socks['stage2R']   )

            x2_left_tensor = torch.Tensor(x2_left).view(1,128,8,8)
            x2_rigt_tensor = torch.Tensor(x2_rigt).view(1,128,8,8)
            x2 = torch.cat( tuple([x2_left_tensor,x2_rigt_tensor]), 1 )
            # print("x2 = ", x2.shape)
            # x2 = x2_left
        if batch_idx >= 1:
            outputs = client_recv( socks['fc']   )

        q_target.put(targets)

        if batch_idx >= 1:
            shifted_target = q_target.get()
            predicted = np.argmax(outputs)
            total   += 1
            correct += predicted==shifted_target
            print(correct,'/',total)

        if args.time_meas:
            print("batch #%d takes %f cycles" % (batch_idx, time.time()-start_time))
