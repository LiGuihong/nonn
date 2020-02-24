import tvm
from tvm import rpc
from tvm.contrib import util, cc, graph_runtime as runtime

from PIL import Image
import numpy as np

######################################################################
# synset is used to transform the label from number of ImageNet class to
# the word human can understand.
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'synset.txt'
with open(synset_name) as f:
    synset = eval(f.read())


######################################################################
# In order to test our model, here we download an image of cat and
# transform its format.
img_name = 'cat.png'
image = Image.open(img_name).resize((224, 224))

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

x = transform_image(image)


ctx = tvm.cpu()
loaded_graph = open("deploy_graph.json").read()
loaded_lib = tvm.module.load("./net.tar")
loaded_params = bytearray(open("deploy_param.params", "rb").read())
input_data = tvm.nd.array(x.astype('float32'))


# create the remote runtime module
module = runtime.create(loaded_graph, loaded_lib, ctx)
# set parameter (upload params to the remote device. This may take a while)
module.load_params(loaded_params)
# run
module.run(data=input_data)
# get output
out = module.get_output(0)
# get top1 result
top1 = np.argmax(out.asnumpy())
print('TVM prediction top-1: {}'.format(synset[top1]))
