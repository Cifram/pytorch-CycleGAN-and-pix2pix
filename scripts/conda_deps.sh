set -ex
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install pytorch torchvision -c pytorch # add cuda90 if CUDA 9
conda install dominate -c conda-forge # install dominate
