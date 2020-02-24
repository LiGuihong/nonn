import struct
import numpy as np

time_start = 0
time_recv_start = 0

def normalize( x, scalar=1/8, offset=0.5 ): return x*scalar+offset
def to_img( x ):
    return normalize(x.transpose(1,2,0))
def ftr_to_image( ftr, height, width=None ):
    width = width if width is not None else height
    ftr = ftr.transpose(2,1,0)
    ftr = ftr.reshape( height, width, -1)
    print("ftr.shape = ", ftr.shape)
    ftr = to_img(ftr).reshape(height, -1)
    return ftr

dlen_table = {'f':4, 'h':2}

def bytestr_to_list( input, dtype='f' ):
    lst = []
    req_len = len(input)

    dlen = dlen_table[dtype]
    lst = struct.unpack( '%s' % int(req_len/dlen) + dtype, input )
    return lst

def list_to_bytestr( input, dtype='f' ):
    byte_flat_data = struct.pack( '%s' % len(input) + dtype, *input)
    return byte_flat_data


def header_asbyte( data ):
    return struct.pack('d',len(data))

def header_decode( header_str ):
    dlen = int(struct.unpack('d',header_str[:8])[0])
    return dlen

def recv( sock, batch_size=512, header_len=8 ):
    import time
    global time_recv_start
    global time_start

    header_str = sock.recv(header_len)
    if len(header_str) is header_len:
        time_recv_start = time.time()
        time_start = time.time()
        show_time("Start a xmission")
        dlen = int(header_decode(header_str))
        request = bytes(0)
        for offset in range( 0, dlen, batch_size ):
            request = request + sock.recv(batch_size)
        while not dlen == len(request):
            print("dlen, request = ", dlen, ", ", len(request), type(dlen), type(len(request)), dlen==len(request), dlen == len(request))
            print("Lose ", dlen-len(request), "/", dlen, " bytes")
            request = request + sock.recv(dlen-len(request))
        show_time("End a " + str(dlen) + "-transmission")
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

def show_time( msg="" ):
    import time
    global time_start
    print("\t%-25s on %.1f" % (msg, ((time.time()-time_start)*1000)), "(%.1f)" % ((time.time()%10)*1000))

def nn_server( model, host, port, i_shape, i_fmt='f', o_fmt='f', is_torch=True, i_xpose=None, o_xpose=None, img_dump=False, img_dump_size=None ):
    import time
    import socket
    if img_dump:
        import matplotlib.pyplot as plt
        _,ax1 = plt.subplots()
        _,ax2 = plt.subplots()

    img_index = 0
    time_infr = []
    time_recv = []
    time_send = []

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
                    print("Img #%d" % img_index)
                    request = recv(c)

                    if not request:
                        import statistics
                        print("[-] Not Received")
                        print("Time", time.time()%100, ": Start %d-byte transmission at " % len(byte_flat_data))
                        print("Input  real  cal: mean = %.3e" % (statistics.mean(time_recv)), "median = %.3e" % statistics.median(time_recv))
                        print("Output real  pkg: mean = %.3e" % (statistics.mean(time_send)), "median = %.3e" % statistics.median(time_send))
                        print("Calculation real: mean = %.3e" % statistics.mean(time_infr), "median = %.3e" % statistics.median(time_infr))
                        time_infr, time_recv, time_send = [], [], []
                        print("--------")
                        return

                    data_list = bytestr_to_list(request,i_fmt)
                    if is_torch:
                        data_tensor = torch.Tensor(data_list).view(i_shape)

                        outputs = model(data_tensor).view(-1).tolist()
                    else:
                        import numpy as np
                        import tvm
                        data_np = np.array(data_list)
                        data_np.resize(i_shape)
                        if i_xpose is not None:
                            data_np = data_np.transpose(i_xpose)
                        data_np = data_np.astype(np.float32)
                        show_time("Input xlation is done")
                        model.set_input("0", tvm.nd.array(data_np))
                        time_recv.append( time.time()-time_recv_start )
                        show_time("Input is ready")

                        time_infr_start = time.time()
                        model.run()
                        time_infr.append( time.time()-time_infr_start )
                        show_time("Inference is done")

                        time_send_start = time.time()
                        outputs = model.get_output(0).asnumpy()
                        if o_xpose is not None:
                            outputs = outputs.transpose(o_xpose)
                        if img_dump:
                            ax1.imshow( ftr_to_image(outputs[0], height=img_dump_size))
                            ax2.imshow( to_img(data_np[0]))
                            plt.draw()
                            plt.pause(1)

                        outputs = outputs.flatten().tolist()

                    byte_flat_data = list_to_bytestr( outputs, o_fmt )

                    byte_flat_data = header_asbyte(byte_flat_data) + byte_flat_data

                    show_time("Output is ready")
                    c.sendall(byte_flat_data)
                    show_time("Output is sent")
                    time_send.append( time.time()-time_send_start )
                    img_index += 1
