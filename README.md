# Model-based Reinforcement Learning for PointGoal Navigation in Habitat

Daniel Mouritzen

[Presentation slides](https://danmou.github.io/presentation.html)

This repository contains the source code for my master's thesis at the [Robotics and Perception Group](http://rpg.ifi.uzh.ch).
The final model, called MerCur-Re (Drea**mer** with **Cur**riculum and **Re**training), is based on
[Dreamer](https://github.com/google-research/dreamer/) by Danijar Hafner and achieves more than 70% success rate in the
Gibson validation scenes after only 1M interactions with the test scenes, given only sparse rewards and monocular RGB input.
See the slides for more results.

### Installation
#### Development
    conda env create -f environment.yml
    conda activate thesis

#### System install
    python setup.py install

### Installation with Habitat-sim
To be able to run habitat-sim locally:

#### Ubuntu 16.04 / Debian Buster
```bash
sudo apt update && sudo apt install libjpeg-dev libpng-dev libglfw3-dev libglm-dev libx11-dev libomp-dev libegl1-mesa-dev
# On Debian additionally install libglvnd-dev

git clone --recurse-submodules https://github.com/facebookresearch/habitat-sim.git
git clone --recurse-submodules git@github.com:uzh-rpg/master_thesis_mouritzen.git thesis

cd thesis
conda env create -f environment.yml
conda activate thesis

cd ../habitat-sim
git checkout v0.1.4  # Use version tag corresponding to installed version of Habitat API
pip install -r requirements.txt
python setup.py install --headless --with-cuda
```

#### Installing SLAM baseline dependencies
Create a new conda env with all the deps in environment.yml installed (SLAM deps are commented out).
Run the script `./project/habitat_baselines/slambased/install_deps.sh`.

### Running
See
```bash
./run.py --help
```
E.g.
```bash
./run.py train --gpus=0 -c gibson_short -c dense_reward -c dreamer
```

#### PPO baseline
```bash
./run.py habitat-baseline --gpus=0 -dc default --run-type train --exp-config ppo_pointnav --num-processes=32 -c sparse_reward -c gibson -m logs/.../checkpoints/ckpt.123.pth
```

#### SLAM baseline
```bash
./run.py evaluate --gpus=0 -n 10 -b slam
```

### Setup MuJoCo and DeepMind Control Suite
```bash
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

conda install git+git://github.com/deepmind/dm_control.git
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

#### Modifying and analyzing datasets using `jq`
Extracting the first episode from dataset and saving as new dataset:
```bash
gzip -dc data/datasets/pointnav/habitat-test-scenes/v1/train/train.json.gz | jq "{episodes: [.episodes[0]]}" | gzip -c > data/datasets/pointnav/habitat-test-scenes/v1/train/single.json.gz
```
Extracting episodes from a specific environment and saving as new dataset:
```bash
split=val; gzip -dc data/datasets/pointnav/habitat-test-scenes/v1/${split}/${split}.json.gz | jq '{episodes: [.episodes[] | select(.scene_id | contains("castle"))]}' | gzip > data/datasets/pointnav/habitat-test-scenes/v1/${split}/${split}_castle.json.gz
```
Extracting specific number of random episodes and saving as new dataset:
```bash
gzip -dc data/datasets/pointnav/habitat-test-scenes/v1/val/val.json.gz | jq '{episodes: [([.episodes[] | select(.scene_id | contains("castle")) | select(.episode_id | tonumber | . % 3 == 0)] | .[0:15])[], ([.episodes[] | select(.scene_id | contains("gogh")) | select(.episode_id | tonumber | . % 3 == 0)] | .[0:15])[]]}' | gzip > data/datasets/pointnav/habitat-test-scenes/v1/val_mini/val_mini.json.gz
```
Generating dataset of only short episodes:
```bash
for F in $(ls data/datasets/pointnav/gibson/v1/train/content/*.json.gz); do
    gzip -dc $F | jq '{episodes: [.episodes[] | select(.info.geodesic_distance < 2.0)]}' | gzip > $(echo $F | sed s/train/train_short/g)
done
cp data/datasets/pointnav/gibson/v1/train/train.json.gz data/datasets/pointnav/gibson/v1/train_short/train_short.json.gz
```
Calculating 5th, 50th and 95th percentiles of geodesic distances:
```bash
split=val; gzip -dc data/datasets/pointnav/gibson/v1/${split}/${split}.json.gz | jq '[.episodes[].info.geodesic_distance] | sort | .[length*(0.05, 0.5, 0.95) | round]'
```
Estimating percentiles for large dataset:
```bash
gzip -dc $(ls data/datasets/pointnav/gibson/v1/train/content/*.json.gz | shuf | head -n 10) | jq '.episodes[].info.geodesic_distance' | jq --slurp 'sort | .[length*(0.05, 0.5, 0.95) | round]'
```