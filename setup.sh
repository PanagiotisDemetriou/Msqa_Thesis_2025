set -euo pipefail
module load CUDA/12.1.1
# Install ninja
conda install ninja

# Install pytorch
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install "numpy<2.0"

# Verify pytorch installation
#python -c "import torch; print(f'{torch.cuda.is_available()=}'); x = torch.rand(5, 3); x = x.cuda(); print(f'{x.device}')"

# install pytorch_geometric (see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
pip install torch_geometric

pip install -r requirements.txt

cd Pointcept-main/libs/pointops
python setup.py install
cd ../../..

# Install flash attention
MAX_JOBS=16 pip install flash-attn --no-build-isolation
