# Biologically Inspired Navigation for Embodied AI

### Installation
#### Development
    conda env create
    conda activate thesis

#### System install
    python setup.py install

### Installation with Habitat-sim
To be able to run habitat-sim locally:

#### Ubuntu 16.04 / Debian Buster
```bash
sudo apt update && sudo apt install libjpeg-dev libpng-dev libglfw3-dev libglm-dev libx11-dev libomp-dev libegl1-mesa-dev
# On Debian additionally install libglvnd-dev
conda create -n thesis python=3.6 cmake=3.14 tensorflow-gpu=1.13.1
conda activate thesis

git clone --recurse-submodules https://github.com/facebookresearch/habitat-sim.git
git clone --recurse-submodules git@github.com:uzh-rpg/master_thesis_mouritzen.git thesis

wget https://www.roboti.us/download/mujoco200_linux.zip
mkdir ~/.mujoco
unzip mujoco200_linux.zip -d ~/.mujoco/
rm mujoco200_linux.zip
wget https://www.roboti.us/getid/getid_linux
chmod +x getid_linux
./getid_linux
# Copy id and generate license at https://www.roboti.us/license.html
# Save license to ~/.mujoco/mjkey.txt
rm getid_linux

cd habitat-sim
pip install -r requirements.txt
python setup.py install --headless --with-cuda

cd ../thesis
git submodule update --init --recursive
python setup.py install
```

### Datasets
```bash
cd <data_dir>

# Test dataset
wget http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip
unzip habitat-test-scenes.zip
mv data/* .
rm -rf data/ habitat-test-scenes.zip

# Replica
wget https://raw.githubusercontent.com/facebookresearch/Replica-Dataset/master/download.sh
# Note: the following command requires the pigz package (can be installed from apt or conda)
bash download.sh scene_datasets/replica
rm replica_v1_0.tar.gz.part?? download.sh
# Note: there is not yet an official task dataset for Replica

# Gibson
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip
mkdir -p datasets/pointnav/gibson/v1
unzip pointnav_gibson_v1.zip -d datasets/pointnav/gibson/v1
rm pointnav_gibson_v1.zip
# Sign agreement and download gibson_habitat_trainval.zip: https://goo.gl/forms/OxAQHbl1v97BJ3Sg1
mkdir -p scene_datasets/
unzip gibson_habitat_trainval.zip -d scene_datasets/
rm gibson_habitat_trainval.zip

# Matterport3d
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip
mkdir -p datasets/pointnav/mp3d/v1
unzip pointnav_mp3d_v1.zip -d datasets/pointnav/mp3d/v1
rm pointnav_mp3d_v1.zip
# Sign agreement and download download_mp.py: https://niessner.github.io/Matterport/
python2 download_mp.py --task habitat -o .
# Note: You only need the habitat zip archive and not the entire Matterport3D dataset.
unzip v1/tasks/mp3d_habitat.zip -x README.txt -d scene_datasets/
rm -rf v1/ download_mp.py
```