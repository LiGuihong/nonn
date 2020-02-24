import struct
import tvm
import tcp_util
import numpy as np
import torch

def nn_server( model, host, port, i_shape, i_fmt='f', o_fmt='f', buf_size=4096 ):
    
    import socket


    # create socket
    while 1:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:

            sock.bind((host, port))
            print("[+] Listening on {0}:{1}".format(host, port))
            sock.listen(5)
            # permit to access
            conn, addr = sock.accept()

            with conn as c:
                # display the current time
                # time = datetime.now().ctime()
                print("[+] Connecting by {0}:{1}".format(addr[0], addr[1]))

                while True:
                    request = c.recv(buf_size)

                    if not request:
                        print("[-] Not Received")
                        break

                    data_list = tcp_util.bytestr_to_list(request,i_fmt)
                    print("data_list[0].type = ", type(data_list[0]))
                    data_tensor = np.reshape( data_list, i_shape )
                    print("data_tensor[0].type = ", data_tensor[0], data_tensor.dtype)


                    model.set_input("0", tvm.nd.array(data_tensor))
                    
                    # Inference
                    model.run()
                    outputs = model.get_output(0, tvm.nd.empty((1,256)))

                    byte_flat_data = tcp_util.list_to_bytestr( outputs.asnumpy().flatten().tolist(), o_fmt )

                    c.sendall(byte_flat_data)
