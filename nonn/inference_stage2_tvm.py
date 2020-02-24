import tvm
import numpy as np
from PIL import Image

if __name__ == "__main__":
    # Arguments setting
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  help="Path of onnx model", type=str,   required=True)
    parser.add_argument("--bsize",  help="Batch size",         type=int,   default=1)
    parser.add_argument("--time_meas", help="Measure the time of each step", action='store_true')
    parser.add_argument("--ip",         help="IP addr",       type=str, default='127.0.0.1')
    parser.add_argument("--port",       help="Port number",   type=int, default=60260)

    args = parser.parse_args()

    if args.time_meas:
        import time
        start_time = time.time()
        temp_time = time.time()

    # Load cross-compiled model
    loaded_json = open(args.model+".json").read()
    loaded_lib = tvm.module.load("./"+args.model+".tar")
    loaded_params = bytearray(open(args.model+".params", "rb").read())

    if args.time_meas:
        print("Load module takes", "%1f" % (time.time()-temp_time), " at ", "%1f" % (time.time()-start_time))
        temp_time = time.time()

    from tvm.contrib import util, cc, graph_runtime
    ctx = tvm.cpu()
    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)

    import torch
    import nn_format
    torch.backends.cudnn.enabled = False

    with torch.no_grad():
        nn_format.nn_server( model=module, host=args.ip, port=args.port, i_shape=(1,3,32,32), is_torch=False, o_xpose=(0,1,4,2,3) )