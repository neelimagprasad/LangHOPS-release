# LangHOPS Diagram and PartImageNet Training Commands

These commands assume the CURC/SSH environment and paths used for the current runs.

## Common Setup

Run this at the start of each shell session:

```bash
cd /projects/nepr1244/LangHOPS-release

conda activate /projects/nepr1244/software/anaconda/envs/langhops

export DETECTRON2_DATASETS=/projects/nepr1244/PartGLEE/datasets
export CUDA_HOME=/curc/sw/cuda/12.1.1
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH

export TMPDIR=/projects/nepr1244/pip_cache/temp
export TORCH_HOME=/projects/nepr1244/pip_cache/torch
mkdir -p "$TMPDIR" "$TORCH_HOME"
```

## Prepare Diagram Exp1 JSONs

Run once before training on diagrams:

```bash
cd /projects/nepr1244/LangHOPS-release
mkdir -p /projects/nepr1244/PartGLEE/datasets/diagram

python3 projects/PartGLEE/tools/diagram_exp1_to_joint_coco.py \
  --input-json /projects/nepr1244/PartGLEE/scripts/diagram-labels-v3-standard.txt \
  --output-dir /projects/nepr1244/PartGLEE/datasets/diagram \
  --old-prefix /projects/nepr1244/PROJECT_DIAGRAM/Diagram_no_labels_v3 \
  --new-prefix /projects/nepr1244/PROJECT_DIAGRAM/Diagram_no_labels_v3
```

Verify:

```bash
ls -lh /projects/nepr1244/PartGLEE/datasets/diagram/
```

Expected files:

```text
diagram_train_joint.json
diagram_val_joint.json
```

## Debug Diagram Exp1 Loading

This verifies the mapper is loading real diagram images and annotations:

```bash
export LANGHOPS_DEBUG_DATASET_SAMPLES=5

python3 projects/PartGLEE/train_net.py \
  --config-file projects/PartGLEE/configs/Training/Diagram-Exp1-Pretrain-RN50_LLM.yaml \
  --num-gpus 1 \
  MODEL.WEIGHTS projects/PartGLEE/checkpoint/PartGLEE_converted_from_GLEE_RN50.pth \
  SOLVER.MAX_ITER 5 \
  SOLVER.CHECKPOINT_PERIOD 999999 \
  OUTPUT_DIR projects/PartGLEE/output/debug/diagram_exp1_debug
```

Turn off debug prints before real runs:

```bash
unset LANGHOPS_DEBUG_DATASET_SAMPLES
```

## Train Diagram Exp1

This trains bbox-only on `diagram_joint_train` to produce a diagram-pretrained checkpoint.

```bash
python3 projects/PartGLEE/train_net.py \
  --config-file projects/PartGLEE/configs/Training/Diagram-Exp1-Pretrain-RN50_LLM.yaml \
  --num-gpus 2 \
  MODEL.WEIGHTS projects/PartGLEE/checkpoint/PartGLEE_converted_from_GLEE_RN50.pth \
  OUTPUT_DIR projects/PartGLEE/output/Training/Diagram-Exp1-Pretrain-RN50/
```

Resume:

```bash
python3 projects/PartGLEE/train_net.py \
  --config-file projects/PartGLEE/configs/Training/Diagram-Exp1-Pretrain-RN50_LLM.yaml \
  --num-gpus 2 \
  --resume \
  OUTPUT_DIR projects/PartGLEE/output/Training/Diagram-Exp1-Pretrain-RN50/
```

## Train PartImageNet Baseline

This is the PartImageNet-only LangHOPS RN50 baseline. It uses mask/dice losses, so it is the comparable segmentation baseline.

```bash
python3 projects/PartGLEE/train_net.py \
  --config-file projects/PartGLEE/configs/Training/PartImageNet-Baseline-RN50_LLM.yaml \
  --num-gpus 2 \
  MODEL.WEIGHTS projects/PartGLEE/checkpoint/PartGLEE_converted_from_GLEE_RN50.pth
```

Resume:

```bash
python3 projects/PartGLEE/train_net.py \
  --config-file projects/PartGLEE/configs/Training/PartImageNet-Baseline-RN50_LLM.yaml \
  --num-gpus 2 \
  --resume \
  OUTPUT_DIR projects/PartGLEE/output/Training/PartImageNet-Baseline-RN50/
```

## Train PartImageNet From Diagram Checkpoint

After diagram Exp1 training finishes, use its checkpoint as `MODEL.WEIGHTS` for PartImageNet fine-tuning:

```bash
python3 projects/PartGLEE/train_net.py \
  --config-file projects/PartGLEE/configs/Training/PartImageNet-Baseline-RN50_LLM.yaml \
  --num-gpus 2 \
  MODEL.WEIGHTS projects/PartGLEE/output/Training/Diagram-Exp1-Pretrain-RN50/model_final.pth \
  OUTPUT_DIR projects/PartGLEE/output/Training/Diagram-Exp1-to-PIN-RN50/
```

## Prepare Diagram Exp2 Part Classification JSONs

Run once before Experiment 2 training. The registration expects the Exp2 JSON and canonical Exp1 labels under `$DETECTRON2_DATASETS/diagram`.

```bash
cd /projects/nepr1244/LangHOPS-release
mkdir -p /projects/nepr1244/PartGLEE/datasets/diagram

cp /projects/nepr1244/PartGLEE/scripts/diagram_parts_v3_exp2.json \
   /projects/nepr1244/PartGLEE/datasets/diagram/

cp /projects/nepr1244/PartGLEE/scripts/diagram-labels-v3-standard.txt \
   /projects/nepr1244/PartGLEE/datasets/diagram/
```

Verify:

```bash
ls -lh /projects/nepr1244/PartGLEE/datasets/diagram/diagram_parts_v3_exp2.json
ls -lh /projects/nepr1244/PartGLEE/datasets/diagram/diagram-labels-v3-standard.txt
```

## Debug Diagram Exp2 Loading

```bash
export LANGHOPS_DEBUG_DATASET_SAMPLES=5

python3 projects/PartGLEE/train_net.py \
  --config-file projects/PartGLEE/configs/Training/Diagram-Exp2-PartClassification-RN50_LLM.yaml \
  --num-gpus 1 \
  MODEL.WEIGHTS projects/PartGLEE/checkpoint/PartGLEE_converted_from_GLEE_RN50.pth \
  SOLVER.MAX_ITER 5 \
  SOLVER.CHECKPOINT_PERIOD 999999 \
  OUTPUT_DIR projects/PartGLEE/output/debug/diagram_exp2_debug
```

Turn off debug prints before real runs:

```bash
unset LANGHOPS_DEBUG_DATASET_SAMPLES
```

## Train Diagram Exp2

This trains the cropped-part classification setup from Experiment 2.

```bash
python3 projects/PartGLEE/train_net.py \
  --config-file projects/PartGLEE/configs/Training/Diagram-Exp2-PartClassification-RN50_LLM.yaml \
  --num-gpus 2 \
  MODEL.WEIGHTS projects/PartGLEE/checkpoint/PartGLEE_converted_from_GLEE_RN50.pth \
  OUTPUT_DIR projects/PartGLEE/output/Training/Diagram-Exp2-PartClassification-RN50/
```

## Train PartImageNet From Exp2 Checkpoint

After Exp2 training finishes, use its checkpoint as initialization for PartImageNet fine-tuning:

```bash
python3 projects/PartGLEE/train_net.py \
  --config-file projects/PartGLEE/configs/Training/PartImageNet-Baseline-RN50_LLM.yaml \
  --num-gpus 2 \
  MODEL.WEIGHTS projects/PartGLEE/output/Training/Diagram-Exp2-PartClassification-RN50/model_final.pth \
  OUTPUT_DIR projects/PartGLEE/output/Training/Diagram-Exp2-to-PIN-RN50/
```

## Evaluate a PartImageNet Checkpoint

Use `--eval-only`. This reports COCO-style `bbox` and `segm` metrics, including AP/AP50/AP75.

```bash
python3 projects/PartGLEE/train_net.py \
  --config-file projects/PartGLEE/configs/Training/PartImageNet-Baseline-RN50_LLM.yaml \
  --num-gpus 1 \
  --eval-only \
  MODEL.WEIGHTS projects/PartGLEE/output/Training/PartImageNet-Baseline-RN50/model_final.pth \
  OUTPUT_DIR projects/PartGLEE/output/Eval/PartImageNet-Baseline-RN50/
```

To evaluate the diagram-to-PIN model:

```bash
python3 projects/PartGLEE/train_net.py \
  --config-file projects/PartGLEE/configs/Training/PartImageNet-Baseline-RN50_LLM.yaml \
  --num-gpus 1 \
  --eval-only \
  MODEL.WEIGHTS projects/PartGLEE/output/Training/Diagram-Exp1-to-PIN-RN50/model_final.pth \
  OUTPUT_DIR projects/PartGLEE/output/Eval/Diagram-Exp1-to-PIN-RN50/
```

## Notes

- `PartImageNet-Baseline-RN50_LLM.yaml` is the correct config for PartImageNet segmentation training. It keeps `MASK_WEIGHT: 5.0` and `DICE_WEIGHT: 5.0`.
- `Diagram-Exp1-Pretrain-RN50_LLM.yaml` is bbox-only and should be used only for diagram Exp1 pretraining.
- If checkpoint saves hit quota issues, delete old checkpoints or old output folders. Training checkpoints include optimizer state and can be much larger than model-only weights.
- `DETECTRON2_DATASETS` must be exported, not just set as a shell variable.
