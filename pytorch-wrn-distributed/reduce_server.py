def recv_as_tensor( sock, shape ):
    import torch
    import nn_format
    return torch.Tensor(nn_format.bytestr_to_list(nn_format.recv(sock))).view(shape)

def send_a_tensor( sock, data ):
    import nn_format
    reduce_msg = nn_format.list_to_bytestr(data.contiguous().view((-1,)).tolist())
    reduce_msg = nn_format.header_asbyte(reduce_msg) + reduce_msg
    sock.sendall( reduce_msg )

import torch
import socket
import nn_format

socks = [None]*2
socks[0] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socks[0].connect( ('127.0.0.1', 5003) )
socks[1] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socks[1].connect( ('127.0.0.1', 5002) )
while 1:
	r1 = recv_as_tensor( socks[0], (1,-1,32,32))
	r2 = recv_as_tensor( socks[1], (1,-1,32,32))
#	merged = torch.cat( tuple([r1]), 1)
	merged = torch.cat( tuple([r1,r2]), 1)
	send_a_tensor( socks[0], merged)
	send_a_tensor( socks[1], merged)


#     header_str = sock.recv(header_len)
#     if len(header_str) is header_len:
#         dlen = nn_format.header_decode(header_str)
#         request = bytes(0)
#         for offset in range( 0, dlen, batch_size ):
#             request = request + sock.recv(batch_size)
#         if dlen is not len(request):
#             request = request + sock.recv(dlen-len(request))
#         assert( dlen == len(request) )
#         return request

# #         response =  socks[0].recv(1024)
# # #        print("response = ", response)
# #         socks[0].sendall(response)

# def recv( sock, batch_size=512, header_len=8 ):
#     header_str = sock.recv(header_len)
