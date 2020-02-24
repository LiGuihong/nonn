import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--onnx_path",  help="Path of onnx model", type=str, required=True)
parser.add_argument("--batch_size", help="Batch size",    type=int, default=1)
parser.add_argument("--to_rasp",    help="Compile to Raspberry", action='store_true')
parser.add_argument("--to_local",   help="Compile to local", action='store_true')
parser.add_argument("--input_size", help="Compile to local", type=str, required=True)
args = parser.parse_args()

import tvm
import onnx
import nnvm
import numpy as np

img_size = [int(x) for x in args.input_size.split(',')]
input_size = tuple( [args.batch_size] + img_size )

model_name = args.onnx_path.split('/')[-1].split('.')[0]
onnx_model = onnx.load_model( args.onnx_path )
sym, params = nnvm.frontend.from_onnx(onnx_model)

import nnvm.compiler
# assume first input name is data
input_name = sym.list_input_names()[0]
shape_dict = {input_name: input_size}
for x in params:
    if params[x].shape == ():
        params[x] = tvm.nd.array(np.float32(0))

# Set cross-compilation target
postfixs = []
targets  = []

if args.to_local:
    postfixs += ['']
    targets += [tvm.target.create('llvm --system-lib')]
    path = "x86/"

if args.to_rasp:
    postfixs += ['-rasp']
    targets += [tvm.target.arm_cpu('rasp3b')]
    path = "rasp/"


# Cross-compile depends on targets
for postfix, target in zip(postfixs, targets):
    with nnvm.compiler.build_config(opt_level=3):
        graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)
    
    lib.export_library( path+model_name+postfix+'.tar', fcompile=False)
    with open( path+model_name+postfix+".json", "w") as fo:
        fo.write(graph.json())
    with open( path+model_name+postfix+".params", "wb") as fo:
        fo.write(nnvm.compiler.save_param_dict(params))
