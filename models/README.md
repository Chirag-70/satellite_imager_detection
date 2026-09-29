# Model package

## Primary change-detection model
**ChangeFormerV6 — LEVIR-CD checkpoint**

Official project:
https://github.com/wgcban/ChangeFormer

Official release referenced by the authors:
https://github.com/wgcban/ChangeFormer/releases/tag/v0.1.0

The authors provide pretrained ChangeFormer checkpoints for LEVIR-CD and
DSIFN-CD. The binary checkpoint is intentionally not fabricated or replaced
with an unrelated model.

Run:

    python models/download_models.py

Then unpack the downloaded release and connect the checkpoint through
`changeformer_adapter.py`.

## Other open-source model families worth evaluating

1. BIT-CD — Transformer-based remote-sensing change detection.
2. SNUNet-CD — Siamese nested U-Net change detection.
3. ChangeFormer — transformer Siamese change detection.
4. U-Net / DeepLabV3+ / SegFormer — building-footprint semantic segmentation.
5. SAM / SAMGeo — interactive object segmentation for analyst review.
6. RemoteCLIP — remote-sensing image/text retrieval and semantic similarity.
7. YOLO/xView — overhead object detection; useful as a separate object layer,
   not as a temporal change detector.

Important:
A model trained on LEVIR-CD is primarily a building-change benchmark model.
It should not be presented as a validated detector of military activity,
installations, vehicles, or other sensitive operational categories without
appropriate data, training and evaluation.
