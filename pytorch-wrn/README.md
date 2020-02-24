# Requirement

- [TVM](https://docs.tvm.ai/install/from_source.html)

# Test model

Once you have a model and pre-trained parameters, you can test and evaluate the accuracy with the following command.
```
python3 inference.py --pt7_path "../model.pt7" 
```

# Convert PyTorch model to x86 or Raspberry

This flow tries to compile a pre-trained Wide-ResNet to different platforms.

## Load model and compile to onnx
This part loads the pre-trained parameters in to a PyTorch model.
```
python3 net.py --to_onnx --pt7_path="../model.pt7" --output "wrn2"
```
You should see a ```wrn.onnx``` in your current directory now. If you don't specify output name by ```--output=<output_name>```, the default name will be "wrn.onnx"

## Cross-compile onnx model to different platform
You can choose the target you want to deploy the model on. For example, the local machine can be selected by adding a ```--to_local``` flag
```
python3 onnx_build.py --onnx_path="wrn2.onnx" --to_local
```
Or you can specify Raspberry Pi3 as your target
```
python3 onnx_build.py --onnx_path="wrn2.onnx" --to_rasp
```

## Deploy a model on your platform
To deploy your model, make sure that your environment has [installed TVM runtime](https://docs.tvm.ai/tutorials/nnvm/deploy_model_on_rasp.html#build-tvm-runtime-on-device).

### Single image deployment
Prior to batched deployment, use a single image first as test.
```
python3 deploy_local_single.py --model wrn2 --img truck.png
```
Or you can use URL as image input
```
python3 deploy_local_single.py --model wrn2 --img_url https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile6.png
```
### Batched deployment (Incompleted)
```
python3 deploy_local.py --model wrn2
```