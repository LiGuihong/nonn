## Split teacher

Terminal 0 (Server)
```
# Make sure your config file is correctly set
# The python code should be executed after all the node (reduce server, teachers, FC) are set
python3 inference.py --config config.xml --cifar ~/Datasets/

```

Terminal 1 (Node reduce server)
```
# Make sure your split teachers are executed
python3 reduce_server.py 
```

Terminal 2 (Node teacher-0)
```
source tvm_config.sh 0 8
python3 inference_stage2.py --pt7_path ~/Downloads/evalWRN2Convert/logs/resnet_40-4_model_notOnVal_saveLast_correct/model.pt7 --port 1248
```

Terminal 3 (Node teacher-1)
```
source tvm_config.sh 0 8
python3 inference_stage2b.py --pt7_path ~/Downloads/evalWRN2Convert/logs/resnet_40-4_model_notOnVal_saveLast_correct/model.pt7 --port 1249
```

Terminal 4 (Node FC)
```
source tvm_config.sh 0 8
python3 inference_fc.py --pt7_path ~/Downloads/evalWRN2Convert/logs/resnet_40-4_model_notOnVal_saveLast_correct/model.pt7  --port 1250
```

config file (config-2s.xml)
```
<config>
        <g0>
                <user_name>pi</user_name>
                <ip>127.0.0.1</ip>
                <port>1230</port>
        </g0>
        <g1>
                <user_name>pi</user_name>
                <ip>127.0.0.1</ip>
                <port>1231</port>
        </g1>
        <fc>
                <ip>127.0.0.1</ip>
                <port>1238</port>
        </fc>
</config>
```

