import struct
import torch

dlen_table = {'f':4, 'h':2}
def bytestr_to_list( input, dtype='f' ):
    lst = []
    req_len = len(input)

    dlen = dlen_table[dtype]
    print("bytestr_to_list( %s, %d )" % (dtype,dlen))
    print("input.len = ", len(input))
    for i in range(0, req_len, dlen):
      lst += struct.unpack( dtype, input[i:i+dlen])
    return lst

def list_to_bytestr( input, dtype='f' ):
    byte_flat_data = bytes()
    for x in input:
        byte_flat_data += struct.pack(dtype, x)
    return byte_flat_data

def header_asbyte( data ):
    return struct.pack('d',len(data))

def header_decode( header_str ):
    dlen = int(struct.unpack('d',header_str[:8])[0])
    return dlen

def recv( sock, batch_size=512, header_len=8 ):
    import time
    header_str = sock.recv(header_len)
    if len(header_str) is header_len:
        start_time = time.time()
        dlen = int(header_decode(header_str))
        request = bytes(0)
        for offset in range( 0, dlen, batch_size ):
            request = request + sock.recv(batch_size)
        while not dlen == len(request):
            print("dlen, request = ", dlen, ", ", len(request), type(dlen), type(len(request)), dlen==len(request), dlen == len(request))
            print("Lose ", dlen-len(request), "/", dlen, " bytes")
            request = request + sock.recv(dlen-len(request))
        print("A ", dlen, "-byte transmission takes ", time.time()-start_time, " seconds")
        assert( dlen == len(request) )
        return request

def recv_as_tensor( sock, shape ):
    raw_bytestr = recv(sock)
    data_tensor = torch.Tensor(bytestr_to_list(raw_bytestr))
    return data_tensor.view(shape)

def send_a_tensor( sock, data ):
    data_list = data.contiguous().view((-1,)).tolist()
    data_bytestr = list_to_bytestr(data_list)
    data_msg = header_asbyte(data_bytestr) + data_bytestr
    sock.sendall( data_msg )

def nn_server( model, host, port, i_shape, i_fmt='f', o_fmt='f' ):
    import socket


    # create socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:

        sock.bind((host, port))
        print("[+] Listening on {0}:{1}".format(host, port))
        while 1:
            sock.listen(5)
            # permit to access
            conn, addr = sock.accept()

            with conn as c:
                # display the current time
                # time = datetime.now().ctime()
                print("[+] Connecting by {0}:{1}".format(addr[0], addr[1]))

                while True:
                    request = recv(c)

                    if not request:
                        print("[-] Not Received")
                        break

                    data_list = bytestr_to_list(request,i_fmt)
                    data_tensor = torch.Tensor(data_list).view(i_shape)

                    outputs = model(data_tensor)
                    _, predicted = outputs.max(1)

                    byte_flat_data = list_to_bytestr( outputs.view(-1).tolist(), o_fmt )

                    byte_flat_data = header_asbyte(byte_flat_data) + byte_flat_data

                    c.sendall(byte_flat_data)
