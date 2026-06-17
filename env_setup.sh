# -----------------------------
# CUDA (for compiling CUDA ops)
# -----------------------------
export CUDA_HOME=/curc/sw/cuda/12.1.1
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH

# -----------------------------
# Project
# -----------------------------
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------
# Conda env
# -----------------------------
# If conda isn't initialized in your shell:
# source ~/miniconda3/etc/profile.d/conda.sh
# or: source ~/anaconda3/etc/profile.d/conda.sh

conda create -n langhops python=3.9.2 -y
conda activate langhops

# (Optional but recommended) make pip installs go into env, not user-site
export PIP_DISABLE_PIP_VERSION_CHECK=1
unset PYTHONPATH

# -----------------------------
# PyTorch (cu121 wheels)
# -----------------------------
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121

# -----------------------------
# Python deps
# -----------------------------
pip install shapely==1.7.1
pip install numpy==1.26.4
pip install Pillow==8.4.0
pip install lvis
pip install scipy
pip install fairscale
pip install einops
pip install tensorboard
pip install opencv-python-headless
pip install xformers==0.0.22.post7
pip install timm
pip install ftfy
pip install transformers==4.48.2
pip install "peft<0.7"
pip install setuptools==75.8.0

# install this repo
pip install -e .

# pycocotools fork (不要再用 --user，装进 env 里更干净)
pip install git+https://github.com/wjf5203/cocoapi.git

# -----------------------------
# (Optional) Download pretrained LM
# -----------------------------
# wget -P projects/PartGLEE/clip_vit_base_patch32/ \
#   https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/GLEE/clip_vit_base_patch32/pytorch_model.bin

# -----------------------------
# Compile Deformable DETR ops
# -----------------------------
cd projects/PartGLEE/partglee/models/pixel_decoder/ops/
python setup.py build install
