conda activate pytorch_latest_p36
pip install kaggle
mkdir .kaggle
cd .kaggle/
wget https://raw.githubusercontent.com/LiGuihong/nonn/master/kaggle.json
cd ..
mkdir imagenet
cd imagenet/
mkdir data
data
cd data/
kaggle competitions download -c imagenet-object-localization-challenge 
unzip imagenet-object-localization-challenge.zip
tar -xvf imagenet_object_localization_patched2019.tar.gz
cd ILSVRC/Data/CLS-LOC/val
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
source valprep.sh
rm valprep.sh
