#!/bin/bash

# ===== CUDA =====
export CUDA_HOME= # ToDo, set the CUDA Home Dir
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH

# ===== Conda init =====
# Adjust this path if your conda is installed elsewhere
source ~/miniconda3/etc/profile.d/conda.sh
# or, for Anaconda:
# source ~/anaconda3/etc/profile.d/conda.sh

conda activate langhops

export TOKENIZERS_PARALLELISM=false

echo "Running langhops_all_data_stage1.sh"

# ===== Paths =====
export DETECTRON2_DATASETS= # ToDo: set the dataset dir

PROJ_DIR= #ToDo: set the project LangHOPS dir
CHECKPOINT= #ToDo: set the check point to evaluate on
CONFIG_FILE=$PROJ_DIR/projects/PartGLEE/configs/Training/Joint-Training-Swin-L_LLM.yaml
SCRIPT=$PROJ_DIR/projects/PartGLEE/train_net.py

cd $PROJ_DIR

OUTPUT_DIR= #ToDo: set the output dir
mkdir -p $OUTPUT_DIR

PORT=tcp://127.0.0.1:49150

# ===== Inference =====
python3 $SCRIPT \
  --config-file $CONFIG_FILE \
  --num-gpus 1 \
  --dist-url $PORT \
  --eval-only \
  MODEL.WEIGHTS $CHECKPOINT \
  DATASETS.TEST '("pascalvoc_joint_val", "partimagenet_renamed_joint_val",)' \
  OUTPUT_DIR $OUTPUT_DIR \
  MODEL.MaskDINO.NUM_OBJECT_QUERIES 200 \
  MODEL.MaskDINO.TOPK_OBJECT_QUERIES 30 \
  MODEL.MaskDINO.NUM_PART_QUERIES 10 \
  MODEL.LLM.LLM_TYPE "google/paligemma2-3b-pt-448" \
  MODEL.LLM.HIDDEN_SIZE 2304 \
  MODEL.LLM.USE_LLM False \
  MODEL.LLM.VLM.IS_VLM True \
  MODEL.LLM.PART_QUERY_MODE "clip_query" \
  MODEL.LLM.PART_CLS_EMBED "learnable" \
  MODEL.LLM.CLIP_EMB_PART_ATTEN_MASK False \
  MODEL.MaskDINO.USE_BOX_RESTRICTIONS False \
  TEST.EVAL_PERIOD 500
