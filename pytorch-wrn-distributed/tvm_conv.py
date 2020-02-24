#!/usr/bin/env python
# coding: utf-8

import tvm_format
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      help="Path of *.pt7", type=str, required=True)
    parser.add_argument("--batch_size", help="Batch size",    type=int, default=1)
    parser.add_argument("--ip",         help="IP addr",       type=str, default='127.0.0.1')
    parser.add_argument("--port",       help="Port number",   type=int, default=60260)
    args = parser.parse_args()

    # Load cross-compiled model
    import tvm
    loaded_json = open(args.model+".json").read()
    loaded_lib = tvm.module.load("./"+args.model+".tar")
    loaded_params = bytearray(open(args.model+".params", "rb").read())

    from tvm.contrib import util, cc, graph_runtime
    ctx = tvm.cpu()
    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)

    tvm_format.nn_server( model=module, host=args.ip, port=args.port, i_shape=(1,3,32,32), buf_size=3*32*32*4 )
