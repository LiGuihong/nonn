import struct

dlen_table = {'f':4, 'h':2}
def bytestr_to_list( input, dtype='f' ):
    lst = []
    req_len = len(input)

    dlen = dlen_table[dtype]
    for i in range(0, req_len, dlen):
      lst += struct.unpack( dtype, input[i:i+dlen])
    return lst

def list_to_bytestr( input, dtype='f' ):
    byte_flat_data = bytes()
    for x in input:
        byte_flat_data += struct.pack(dtype, x)
    return byte_flat_data