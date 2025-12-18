# Create conda environment
conda create -n ptv3 python=3.10 -y
source /lustreFS/data/vcg/pdemetriou/miniconda3/etc/profile.d/conda.sh
conda activate ptv3

# Download cuda 12.1
export NVIDIA_DRIVER_FILE=cuda_12.1.0_530.30.02_linux.run
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/${NVIDIA_DRIVER_FILE}

# Locally install cuda 12.1
export LUSTRE_CUDA_PATH=/lustreFS/data/vcg/pdemetriou
export CUDA_VER=12.1
bash ${NVIDIA_DRIVER_FILE} --silent --override --toolkit --toolkitpath=${LUSTRE_CUDA_PATH}/cuda-${CUDA_VER}
#unset LUSTRE_CUDA_PATH
  
# Create activate.sh and add the following bash commands
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo '#!/bin/sh
export LUSTRE_CUDA_PATH=/lustreFS/data/vcg/pdemetriou
export INIT_CUDADIR=$CUDADIR
export INIT_PATH=$PATH
export INIT_NUMBAPRO_NVVM=$NUMBAPRO_NVVM
export INIT_NUMBAPRO_LIBDEVICE=$NUMBAPRO_LIBDEVICE
export INIT_NVCCDIR=$NVCCDIR
export INIT_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export INIT_CPATH=$CPATH
export INIT_CUDA_HOME=$CUDA_HOME
export INIT_CUDA_BIN_PATH=$CUDA_BIN_PATH
cuda=cuda-12.1
export CUDADIR=$LUSTRE_CUDA_PATH/$cuda
export PATH=$CUDADIR/bin:$PATH
export NUMBAPRO_NVVM=$CUDADIR/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=$CUDADIR/nvvm/libdevice/
export NVCCDIR=$CUDADIR/bin/nvcc
export LD_LIBRARY_PATH=$CUDADIR/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDADIR/include:$CPATH
export CUDA_HOME=$CUDADIR
export CUDA_BIN_PATH=$CUDADIR' > $CONDA_PREFIX/etc/conda/activate.d/activate.sh

# Verify cuda installation
conda deactivate
conda activate ptv3
nvcc --version

# Clean cuda installation files
rm -v ${NVIDIA_DRIVER_FILE}
unset NVIDIA_DRIVER_FILE

# Create deactivate.sh and add the following commands
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo '#!/bin/sh
export CUDADIR=$INIT_CUDADIR
export PATH=$INIT_PATH
export NUMBAPRO_NVVM=$INIT_NUMBAPRO_NVVM
export NUMBAPRO_LIBDEVICE=$INIT_NUMBAPRO_LIBDEVICE
export NVCCDIR=$INIT_NVCCDIR
export LD_LIBRARY_PATH=$INIT_LD_LIBRARY_PATH
export CPATH=$INIT_CPATH
export CUDA_HOME=$INIT_CUDA_HOME
export CUDA_BIN_PATH=$INIT_CUDA_BIN_PATH
unset cuda
unset INIT_CUDADIR
unset INIT_PATH
unset INIT_NUMBAPRO_NVVM
unset INIT_NUMBAPRO_LIBDEVICE
unset INIT_NVCCDIR
unset INIT_LD_LIBRARY_PATH
unset INIT_CPATH
unset INIT_CUDA_HOME
unset INIT_CUDA_BIN_PATH
unset LUSTRE_CUDA_PATH' > $CONDA_PREFIX/etc/conda/deactivate.d/deactivate.sh

# Install cudnn (Prometheus cluster)
pip install gdown
gdown 1U8b_ecKwQOpJZiXMtBo9swpu7_Npp0Zi
tar -xvf cudnn-linux-x86_64-8.9.2.26_cuda12-archive.tar.xz
cp -v cudnn-*-archive/include/cudnn*.h $LUSTRE_CUDA_PATH/cuda-${CUDA_VER}/include 
cp -Pv cudnn-*-archive/lib/libcudnn* $LUSTRE_CUDA_PATH/cuda-${CUDA_VER}/lib64 
chmod a+r $LUSTRE_CUDA_PATH/cuda-${CUDA_VER}/include/cudnn*.h $LUSTRE_CUDA_PATH/cuda-${CUDA_VER}/lib64/libcudnn*

# Clean cudnn installation files
rm -v cudnn-linux-x86_64-8.9.2.26_cuda12-archive.tar.xz
rm -rv cudnn-linux-x86_64-8.9.2.26_cuda12-archive/
unset CUDA_VER
unset LUSTRE_CUDA_PATH

# Install ninja
conda install ninja

# Install pytorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2.0"

# Verify pytorch installation
srun -c 1 --gres=gpu:1 python -c "import torch; print(f'{torch.cuda.is_available()=}'); x = torch.rand(5, 3); x = x.cuda(); print(f'{x.device}')"

# install pytorch_geometric (see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch_geometric

pip install -r requirements.txt

cd Pointcept_main/libs/pointops
srun -c 6 --mem=144G --gres=gpu:1 --pty /bin/bash
conda activate ptv3
python setup.py install
exit
cd ../../..
