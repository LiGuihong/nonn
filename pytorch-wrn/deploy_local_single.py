import tvm
import numpy as np
from PIL import Image


# Arguments setting
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model",  help="Path of onnx model", type=str,   required=True)
parser.add_argument("--img",    help="Path of image",      type=str)
parser.add_argument("--img_url",help="URL of image",      type=str)
parser.add_argument("--labels", help="Path of labels",     type=str,   default="batches.meta")
parser.add_argument("--bsize",  help="Batch size",         type=int,   default=1)
parser.add_argument("--means",  help="Means shift",        type=float, nargs='+', default=[125.3, 123.0, 113.9])
parser.add_argument("--std",    help="Means shift",        type=float, nargs='+', default=[63.0, 62.1, 66.7])
parser.add_argument("--time_meas", help="Measure the time of each step", action='store_true')
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

if args.time_meas:
    print("Load param takes", "%1f" % (time.time()-temp_time), " at ", "%1f" % (time.time()-start_time))
    temp_time = time.time()


# Single image input
def transform_image(image):
    image = np.array(image) - np.array(args.means)
    image /= np.array(args.std)
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


if args.img_url is not None:
    import requests
    from io import BytesIO
    response = requests.get(args.img_url)
    image = Image.open(BytesIO(response.content))

elif args.img is not None:
    image = Image.open(args.img)

x = transform_image(image.resize((32,32))).astype(np.float32)
module.set_input("0", tvm.nd.array(x))

if args.time_meas:
    print("Load image takes", "%1f" % (time.time()-temp_time), " at ", "%1f" % (time.time()-start_time))
    temp_time = time.time()


# Inference
module.run()
out_shape = ( args.bsize, 10 )
out = module.get_output(0, tvm.nd.empty(out_shape))

if args.time_meas:
    print("Inference takes", "%1f" % (time.time()-temp_time), " at ", "%1f" % (time.time()-start_time))
    temp_time = time.time()


# Load labels
import pickle
with open( args.labels, "rb") as fo:
    labels = pickle.load(fo)['label_names']


# get output
out_label = np.argmax(out.asnumpy(), axis=1)[0]
print("out = ", labels[out_label])
