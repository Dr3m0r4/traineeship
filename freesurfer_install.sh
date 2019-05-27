#!/bin/bash

wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/6.0.0/freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz

sudo tar -C /usr/local -xzvf freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz

cat export FREESURFER_HOME=/usr/local/freesurfer >> ~/.bashrc
cat source $FREESURFER_HOME/SetUpFreeSurfer.sh >> ~/.bashrc
source ~/.bashrc

rm freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz

mkdir freesurfer.tmp && cd freesurfer.tmp
curl https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/dev/freesurfer-linux-centos6_x86_64-dev.tar.gz -o fsdev.tar.gz
tar -xzvf fsdev.tar.gz

sudo rm -rf ${FREESURFER_HOME}/lib/qt
sudo cp -r freesurfer/lib/qt ${FREESURFER_HOME}/lib/qt
sudo cp freesurfer/bin/freeview freesurfer/bin/qt.conf ${FREESURFER_HOME}/bin/

cd .. && rm -rf freesurfer.tmp
