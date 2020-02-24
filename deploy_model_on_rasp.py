"""
.. _tutorial-deploy-model-on-rasp:

Deploy the Pretrained Model on Raspberry Pi
===========================================
**Author**: `Ziheng Jiang <https://ziheng.org/>`_

This is an example of using NNVM to compile a ResNet model and deploy
it on raspberry pi.
"""

import tvm
import nnvm.compiler
import nnvm.testing
from tvm import rpc
from tvm.contrib import util, cc, graph_runtime as runtime

######################################################################
# .. _build-tvm-runtime-on-device:
#
# Build TVM Runtime on Device
# ---------------------------
#
# The first step is to build tvm runtime on the remote device.
#
# .. note::
#
#   All instructions in both this section and next section should be
#   executed on the target device, e.g. Raspberry Pi. And we assume it
#   has Linux running.
# 
# Since we do compilation on local machine, the remote device is only used
# for running the generated code. We only need to build tvm runtime on
# the remote device.
#
# .. code-block:: bash
#
#   git clone --recursive https://github.com/dmlc/tvm
#   cd tvm
#   make runtime -j4
#
# After building runtime successfully, we need to set environment varibles
# in :code:`~/.bashrc` file. We can edit :code:`~/.bashrc`
# using :code:`vi ~/.bashrc` and add the line below (Assuming your TVM 
# directory is in :code:`~/tvm`):
#
# .. code-block:: bash
#
#   export PYTHONPATH=$PYTHONPATH:~/tvm/python
#
# To update the environment variables, execute :code:`source ~/.bashrc`.

######################################################################
# Set Up RPC Server on Device
# ---------------------------
# To start an RPC server, run the following command on your remote device
# (Which is Raspberry Pi in our example).
#
#   .. code-block:: bash
#
#     python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090
#
# If you see the line below, it means the RPC server started
# successfully on your device.
#
#    .. code-block:: bash
#
#      INFO:root:RPCServer: bind to 0.0.0.0:9090
#

######################################################################
# Prepare the Pre-trained Model
# -----------------------------
# Back to the host machine, which should have a full TVM installed (with LLVM).
# 
# We will use pre-trained model from
# `MXNet Gluon model zoo <https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html>`_.
# You can found more details about this part at tutorial :ref:`tutorial-from-mxnet`.

from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.utils import download
from PIL import Image
import numpy as np

# one line to get the model
block = get_model('resnet18_v1', pretrained=True)

######################################################################
# Now we would like to port the Gluon model to a portable cutational graph.
# It's as easy as several lines.

# We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon
net, params = nnvm.frontend.from_mxnet(block)
# we want a probability so add a softmax operator
net = nnvm.sym.softmax(net)

######################################################################
# Here are some basic data workload configurations.
batch_size = 1
num_classes = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape

######################################################################
# Compile The Graph
# -----------------
# To compile the graph, we call the :any:`nnvm.compiler.build` function
# with the graph configuration and parameters. However, You cannot to
# deploy a x86 program on a device with ARM instruction set. It means
# NNVM also needs to know the compilation option of target device,
# apart from arguments :code:`net` and :code:`params` to specify the
# deep learning workload. Actually, the option matters, different option
# will lead to very different performance.

######################################################################
# If we run the example on our x86 server for demonstration, we can simply
# set it as :code:`llvm`. If running it on the Raspberry Pi, we need to
# specify its instruction set. Set :code:`local_demo` to False if you want
# to run this tutorial with a real device.

local_demo = False

if local_demo:
    target = tvm.target.create('llvm --system-lib')
else:
    target = tvm.target.arm_cpu('rasp3b')
    # The above line is a simple form of
    # target = tvm.target.create('llvm -devcie=arm_cpu -model=bcm2837 -target=armv7l-linux-gnueabihf -mattr=+neon')

with nnvm.compiler.build_config(opt_level=3):
    graph, lib, params = nnvm.compiler.build(
        net, target, shape={"data": data_shape}, params=params)

# After `nnvm.compiler.build`, you will get three return values: graph,
# library and the new parameter, since we do some optimization that will
# change the parameters but keep the result of model as the same.

# Save the library at local temporary directory.
lib.export_library("net.tar", fcompile=False)
with open("deploy_graph.json", "w") as fo:
    fo.write(graph.json())
with open("deploy_param.params", "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))

# module = runtime.create(loaded_graph, loaded_lib, ctx)
# module.load_params(loaded_params)
# module.run(data=input_data)
# out = module.get_output(0).asnumpy()
# print("out.size = ", np.shape(out), out[:10])

