# PN-Cache Test-Time Adaptation (No Backprop) for MobileNetV3
See config-driven `tta_mobilenetv3_pncache.py`. Includes finetune & eval scripts.

## Usage
- Finetune: `python train_finetune.py --config config_train.yaml`
- Evaluate baseline vs TTA: `python eval_tta.py --config config_eval.yaml`
