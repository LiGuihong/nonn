export TVM_HOME="/home/pi/tvm"
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH}
export TVM_BIND_THREADS=$1
export TVM_NUM_THREADS=$2
echo "TVM_BIND_THREADS = " $TVM_BIND_THREADS
echo "TVM_NUM_THREADS  = " $TVM_NUM_THREADS

