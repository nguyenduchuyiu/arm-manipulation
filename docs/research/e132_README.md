# E132 — original SAM 3 text-prompt segmentation in BF16

Status: COMPLETE.

Request: zero-shot segment the pick objects with simple text prompts instead
of task IDs. Use original official facebook/sam3, pinned model revision
3c879f39826c281e95690f02c7821c4de09afae7, through its documented Hugging Face
Transformers implementation. No finetuning or policy architecture changes.
Original model access is gated; existing HF credential was denied access on
2026-10-23. User will report when the account is approved. No unofficial
weight mirror or substitute model is used.

Official sources:
https://github.com/facebookresearch/sam3
https://huggingface.co/facebook/sam3
https://huggingface.co/docs/transformers/model_doc/sam3

Isolated .venv-sam3: Python3.11.16, Transformers5.20.2,HF Hub2.3.0,
tokenizers0.23.3,safetensors0.8.0; shares the existing torch2.7.0+cu128 and
torchvision0.22.0+cu128 via a site-path file. The original .venv is unchanged.
This uses the documented Transformers integration, not the native repository's
Python3.12 installation. Imports succeed and RTX3090 reports BF16 support.
Load all model parameters in BF16 and run CUDA BF16 autocast. No quantization,
optimizer,batch training,epochs,LR,FM,action chunks or physical rollouts.

Pilot:4validation layouts32-35, one source target episode per layout balancing
Red/Blue/Green/Yellow, first available permutation. Four original front RGB
frames/episode:home, last approach, last engage,last lift.16images x4text
queries=64queries, native320x240. Fixed prompts in dataset color order:
red cube, blue cube, green cube, yellow cube. RGB and text only; no GT point,
box,mask,depth or robot state enters SAM3. GT instance masks score outputs.
This small pilot is not a full-data or all-OOD evaluation.

Use official processor/native model preprocessing. Cache vision features once
per image and text embeddings once per prompt, with first-query full-forward
parity check. Threshold confidence0.5,mask probability0.5. With multiple returned
instances, select highest model confidence only; save all proposals. Report
no-detection/multiple-detection counts,mean visible-target IoU,precision/recall,
identity correctness and IoU>=0.5/0.75. GT queries with<8visible pixels are
excluded from visible-mask/identity aggregates and counted separately.

Output:/workspace/robotics_assets/experiments/e132_sam.
Prepare offline inputs: `.venv-sam3/bin/python -m OATFlow.task_mask.sam3_evaluate prepare`.
After HF approval: `.venv-sam3/bin/python -u -m OATFlow.task_mask.sam3_evaluate run`.
The run downloads only official safetensors/config/tokenizer files (approximately
3.44GB for the weight); excludes duplicate sam3.pt. Model cache path:
/workspace/robotics_assets/pretrained/sam3_hf. Run inference in tmux with tee.
Prepared inputs/data hashes and environment versions are recorded; model
inference remains NOT RUN until authorized weights are available.

Results:
{
  "status": "complete",
  "images": 16,
  "queries": 64,
  "visible_queries": 64,
  "mean_IoU": 0.9301741664290225,
  "identity_accuracy": 0.984375,
  "IoU_ge50": 0.984375,
  "IoU_ge75": 0.984375,
  "no_detections": 1,
  "multiple_detections": 1,
  "peak_allocated_GiB": 2.1928982734680176,
  "vision_dtype": [
    "torch.bfloat16"
  ],
  "mask_logits_dtype": [
    "torch.bfloat16"
  ],
  "mean_vision_ms": 151.64125646697357,
  "mean_prompt_head_ms": 34.07886583299842,
  "by_stage": {
    "home": {
      "queries": 16,
      "mean_IoU": 0.9483170568102768,
      "identity_accuracy": 1.0
    },
    "approach": {
      "queries": 16,
      "mean_IoU": 0.9484511403404444,
      "identity_accuracy": 1.0
    },
    "engage": {
      "queries": 16,
      "mean_IoU": 0.882317329550208,
      "identity_accuracy": 0.9375
    },
    "lift": {
      "queries": 16,
      "mean_IoU": 0.9416111390151609,
      "identity_accuracy": 1.0
    }
  },
  "by_prompt": {
    "red cube": {
      "queries": 16,
      "mean_IoU": 0.9577050493274248,
      "identity_accuracy": 1.0
    },
    "blue cube": {
      "queries": 16,
      "mean_IoU": 0.8906067817820347,
      "identity_accuracy": 0.9375
    },
    "green cube": {
      "queries": 16,
      "mean_IoU": 0.9386066166915722,
      "identity_accuracy": 1.0
    },
    "yellow cube": {
      "queries": 16,
      "mean_IoU": 0.9337782179150582,
      "identity_accuracy": 1.0
    }
  }
}
