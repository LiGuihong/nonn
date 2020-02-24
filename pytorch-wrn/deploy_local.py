import tvm
import onnx
import nnvm
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name",  help="Path of onnx model", type=str, required=True)
parser.add_argument("--batch_size", help="Batch size",    type=int, default=1)
args = parser.parse_args()

loaded_json = open(args.model_name+".json").read()
loaded_lib = tvm.module.load("./"+args.model_name+".tar")
loaded_params = bytearray(open(args.model_name+".params", "rb").read())

from tvm.contrib import util, cc, graph_runtime
ctx = tvm.cpu()
module = graph_runtime.create(loaded_json, loaded_lib, ctx)
module.load_params(loaded_params)

# Create the dataset iterator
import cvtransforms
testloader = cvtransforms.create_iterator( False, False, args.batch_size )

# create module
from tvm.contrib import graph_runtime
#module = graph_runtime.create(graph, lib, ctx)

total = 0
correct = 0
for batch_idx, (inputs, targets) in enumerate(testloader):
    # set input and parameters
    module.set_input("0", tvm.nd.array(inputs.numpy()))

    # module.set_input(np.array(inputs))
    module.run()

    # get output
    out_shape = ( args.batch_size, 10 )
    out = module.get_output(0, tvm.nd.empty(out_shape))

    # Accuracy
    total += targets.size(0)
    correct += 1 if np.argmax(out.asnumpy(), axis=1)[0]==int(targets) else 0
    print(correct,'/',total)
    # print("#", batch_idx, ": output = ", np.argmax(out.asnumpy(), axis=1)[0], ", label = ", int(targets) )
