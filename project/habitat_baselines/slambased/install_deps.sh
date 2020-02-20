#!/usr/bin/env bash

# Note: run from project root, i.e. ./project/habitat_baselines/slambased/install_deps.sh

export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig/"
export CPATH="$CONDA_PREFIX/include"
export LIBRARY_PATH="$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
DIR1=$(pwd)
MAINDIR=$(pwd)/3rdparty
mkdir "$MAINDIR"
cd "$MAINDIR" || exit
rm -rf Pangolin ORB_SLAM2 ORB_SLAM2-PythonBindings
git clone https://github.com/stevenlovegrove/Pangolin.git
git clone https://github.com/ducha-aiki/ORB_SLAM2
git clone https://github.com/ducha-aiki/ORB_SLAM2-PythonBindings
mkdir Pangolin/build
cd Pangolin/build || exit
cmake .. -DCMAKE_PREFIX_PATH="$CONDA_PREFIX/" -DCMAKE_INSTALL_PREFIX="$MAINDIR/pangolin_installed"
cmake --build .
cd "$MAINDIR/ORB_SLAM2" || exit
sed -i "s,cmake .. -DCMAKE_BUILD_TYPE=Release,cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$CONDA_PREFIX/ -DCMAKE_INSTALL_PREFIX=$MAINDIR/ORBSLAM2_installed ,g" build.sh
./build.sh
cd build || exit
make install
cd "$MAINDIR/ORBSLAM2_installed/lib" || exit
cp libDBoW2.so libg2o.so "$CONDA_PREFIX/lib/"
mkdir "$MAINDIR/ORB_SLAM2-PythonBindings/build"
cd "$MAINDIR/ORB_SLAM2-PythonBindings/build" || exit
sed -i "s,lib/python3.5/dist-packages,$CONDA_PREFIX/lib/python3.7/site-packages/,g" ../CMakeLists.txt
sed -i "s,PythonLibs 3.5,PythonLibs 3.7,g" ../CMakeLists.txt
sed -i "s,python-py35,python37,g" ../CMakeLists.txt
cmake .. \
-DCMAKE_PREFIX_PATH="$CONDA_PREFIX/" \
-DCMAKE_SHARED_LINKER_FLAGS="-Wl,--no-undefined" \
-DORB_SLAM2_DIR="$MAINDIR/ORBSLAM2_installed" \
-DCMAKE_INSTALL_PREFIX="$MAINDIR/pyorbslam2_installed"
make
make install
cp "$MAINDIR/ORBSLAM2_installed/lib/lib"{DBoW2,g2o}.so "$CONDA_PREFIX/lib/"
cp "$MAINDIR/ORB_SLAM2/Vocabulary/ORBvoc.txt" "${DIR1}/project/habitat_baselines/slambased/data/"
