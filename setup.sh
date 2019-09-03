#!/bin/bash

# download and install miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

/bin/bash ~/miniconda.sh -bf

rm -rf ~/miniconda.sh

# activate conda
source ~/miniconda3/bin/activate
conda init
source ~/.bashrc

conda env create -f environment.yml

conda activate mura
