# Python3 
apt-get python3

# Pip
apt-get install -y python3-pip

# Git
apt-get install -y git

# OpenCV
apt-get install -y build-essential 
apt-get install -y cmake
apt-get install -y libgtk2.0-dev 
apt-get install -y pkg-config 
apt-get install -y libavcodec-dev 
apt-get install -y libavformat-dev 
apt-get install -y libswscale-dev

git clone https://github.com/Itseez/opencv.git
cp -r opencv/ opencv-3/
rm -r opencv/
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ../opencv-3
make 
make install