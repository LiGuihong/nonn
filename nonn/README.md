# Requirement

- [TVM](https://docs.tvm.ai/install/from_source.html)

# Test model

Once you have a model and pre-trained parameters, you can test and evaluate the accuracy with the following command.
```
python3 wrn_onnx_gen.py --pt7_path /home/chingyi/Downloads/NoNN_fixFLOPS_RPI_models/logs/ST_swrn_v6a_WRN40-4_b\=10K_a\=0.9_norm_2S_fixFLOPS_run4/model.pt7 --cifar ../pytorch-wrn --inference
```

# Convert PyTorch model to x86 or Raspberry

This flow tries to compile a pre-trained Wide-ResNet to different platforms.

## Load model and compile to onnx
This part loads the pre-trained parameters in to a PyTorch model.
```
python3 wrn_onnx_gen.py --pt7_path /home/chingyi/Downloads/NoNN_fixFLOPS_RPI_models/logs/ST_swrn_v6a_WRN40-4_b\=10K_a\=0.9_norm_2S_fixFLOPS_run4/model.pt7 --to_onnx
```
You should see ```wrn_g1.onnx```, ```wrn_g2.onnx``` and ```wrn_fc.onnx``` in your current directory now.

## Cross-compile onnx model to different platform
You can choose the target you want to deploy the model on. For example, the local machine can be selected by adding a ```--to_local``` flag
```
python3 onnx_build.py --onnx_path wrn_g1.onnx  --to_local --input_size 3,32,32
python3 onnx_build.py --onnx_path wrn_g2.onnx  --to_local --input_size 3,32,32
python3 onnx_build.py --onnx_path wrn_fc.onnx  --to_local --input_size 80,8,8

```
Or you can specify Raspberry Pi3 as your target
```
python3 onnx_build.py --onnx_path wrn_g1.onnx  --to_rasp --input_size 3,32,32
python3 onnx_build.py --onnx_path wrn_g2.onnx  --to_rasp --input_size 3,32,32
python3 onnx_build.py --onnx_path wrn_fc.onnx  --to_rasp --input_size 80,8,8
```

## Deploy a model on your platform
To deploy your model, make sure that your environment has [installed TVM runtime](https://docs.tvm.ai/tutorials/nnvm/deploy_model_on_rasp.html#build-tvm-runtime-on-device).

### If you want to deploy your model on your x86 machine
```
python3 inference_tvm.py --model x86/wrn_g1 --port 1235 --ip "127.0.0.1" --i_size 3,32,32 --o_xpose=0,1,4,2,3
python3 inference_tvm.py --model x86/wrn_g2 --port 1236 --ip "127.0.0.1" --i_size 3,32,32 --o_xpose=0,1,4,2,3
python3 inference_tvm.py --model x86/wrn_fc --port 1237 --ip "127.0.0.1" --i_size 80,8,8
```

### If you want to deploy your model on your raspberry pi
python3 inference_tvm.py --model x86/wrn_g1 --port 1235 --ip "127.0.0.1" --i_size 3,32,32
python3 inference_tvm.py --model x86/wrn_g2 --port 1236 --ip "127.0.0.1" --i_size 3,32,32
python3 inference_tvm.py --model x86/wrn_fc --port 1237 --ip "127.0.0.1" --i_size 80,8,8
