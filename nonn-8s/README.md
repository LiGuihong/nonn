## Teacher inference

Terminal 0 (Server)
```
# Make sure your config file is correctly set
# The python code should be executed after all the node are set
python3 inference-teacher.py --config config-teacher.xml --time_meas
```

Terminal 1 (Node extraction)
```
source tvm_config.sh 0 8
python3 inference_tvm.py --model x86/wrn_et --i_size 256,8,8 --ip 127.0.0.1 --port 1237
```

Terminal 2 (Node FC)
```
source tvm_config.sh 0 8
python3 inference_tvm.py --model x86/wrn_et --i_size 3,32,32 --ip 127.0.0.1 --port 1236
```

config file (config-teacher.xml)
```
<config>
        <g0>
                <ip>127.0.0.1</ip>
                <port>1236</port>
        </g0>
        <fc>
                <ip>127.0.0.1</ip>
                <port>1237</port>
        </fc>
</config>
```

## 2S

Terminal 0 (Server)
```
# Make sure your config file is correctly set
# The python code should be executed after all the node are set
python3 inference-teacher.py --config config-2s.xml --time_meas
```

Terminal 1 (Node stu1)
```
source tvm_config.sh 0 8
python3 inference_tvm.py --model x86/wrn_g1 --i_size 3,32,32 --ip 127.0.0.1 --port 1230 --time_meas
```

Terminal 2 (Node stu2)
```
source tvm_config.sh 0 8
python3 inference_tvm.py --model x86/wrn_g2 --i_size 3,32,32 --ip 127.0.0.1 --port 1231 --time_meas
```

Terminal 3 (Node FC)
```
source tvm_config.sh 0 8
python3 inference_tvm.py --model ../nonn/x86/wrn_fc --i_size 80,8,8 --ip 127.0.0.1 --port 1238 --time_meas
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
