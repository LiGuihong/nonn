#!/usr/bin/env python
# coding: utf-8

import nn_format
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt7_path",   help="Path of *.pt7", type=str, required=True)
    parser.add_argument("--batch_size", help="Batch size",    type=int, default=1)
    parser.add_argument("--ip",         help="IP addr",       type=str, default='127.0.0.1')
    parser.add_argument("--port",       help="Port number",   type=int, default=60260)
    args = parser.parse_args()

    import torch
    import net

    torch.backends.cudnn.enabled = False
    param = torch.load( args.pt7_path, map_location='cpu' )
    sub_model = net.WRN_extract( batch_size=args.batch_size, weights=param['params'], stats=param['stats'], reduce_port=5003, segment=2, id=0 )

    sub_model.set_requires_grad(False)
    sub_model.eval()

    print(sub_model)

    with torch.no_grad():
        nn_format.nn_server( model=sub_model, host=args.ip, port=args.port, i_shape=(1,3,32,32) )