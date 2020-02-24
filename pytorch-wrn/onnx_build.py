import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--onnx_path",  help="Path of onnx model", type=str, required=True)
parser.add_argument("--batch_size", help="Batch size",    type=int, default=1)
parser.add_argument("--to_rasp",    help="Compile to Raspberry", action='store_true')
parser.add_argument("--to_local",   help="Compile to local", action='store_true')
args = parser.parse_args()

import cvtransforms as dataset
import tvm
import onnx
import nnvm
import numpy as np

model_name = args.onnx_path.split('/')[-1].split('.')[0]
print("model_name = ", model_name)
onnx_model = onnx.load_model( args.onnx_path )
sym, params = nnvm.frontend.from_onnx(onnx_model)

import nnvm.compiler
# assume first input name is data
input_name = sym.list_input_names()[0]
shape_dict = {input_name: (1,3,32,32)}
for x in params:
    if params[x].shape == ():
        params[x] = tvm.nd.array(np.float32(0))

# Set cross-compilation target
postfixs = []
targets  = []

if args.to_local:
    postfixs += ['']
    targets += [tvm.target.create('llvm --system-lib')]

if args.to_rasp:
    postfixs += ['-rasp']
    targets += [tvm.target.arm_cpu('rasp3b')]


# Cross-compile depends on targets
for postfix, target in zip(postfixs, targets):
    with nnvm.compiler.build_config(opt_level=3):
        graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)
    
    lib.export_library( model_name+postfix+'.tar', fcompile=False)
    with open( model_name+postfix+".json", "w") as fo:
        fo.write(graph.json())
    with open( model_name+postfix+".params", "wb") as fo:
        fo.write(nnvm.compiler.save_param_dict(params))
