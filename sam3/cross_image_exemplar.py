# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
Cross-Image Exemplar Search for SAM3

Extract exemplar tokens from reference image, apply to any target image.
Reuses Sam3Processor for preprocessing/postprocessing.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.geometry_encoders import Prompt
from sam3.model import box_ops
from sam3.model.data_misc import interpolate


@dataclass
class SegmentationResult:
    masks: torch.Tensor      # (N, H, W)
    boxes: torch.Tensor      # (N, 4) xyxy, pixel coords
    scores: torch.Tensor     # (N,)
    presence_score: float


class CrossImageExemplarSearch:
    """Extract exemplar from one image, find matches in others."""

    def __init__(self, model, resolution=1008, device="cuda", threshold=0.3):
        self.processor = Sam3Processor(model, resolution, device, threshold)
        self.model = model
        self.device = device
        self.threshold = threshold
        self._tokens = None
        self._mask = None

    @property
    def has_exemplar(self) -> bool:
        return self._tokens is not None

    @torch.inference_mode()
    def set_reference(self, image, bbox: List[float], label: int = 1):
        """Extract and cache exemplar tokens from reference image + bbox."""
        # Use processor for preprocessing + backbone
        state = self.processor.set_image(image)
        h, w = state["original_height"], state["original_width"]

        # Convert xyxy to normalized cxcywh using SAM3's box_ops
        bbox_t = torch.tensor(bbox, device=self.device, dtype=torch.float32)
        bbox_cxcywh = box_ops.box_xyxy_to_cxcywh(bbox_t)
        bbox_cxcywh = bbox_cxcywh / torch.tensor([w, h, w, h], device=self.device)
        box_norm = bbox_cxcywh.view(1, 1, 4)
        prompt = Prompt(
            box_embeddings=box_norm.permute(1, 0, 2),
            box_mask=torch.zeros(1, 1, device=self.device, dtype=torch.bool),
            box_labels=torch.tensor([[label]], device=self.device).permute(1, 0),
        )

        # Get features and encode geometry
        img_ids = self.processor.find_stage.img_ids
        _, feats, pos, sizes = self.model._get_img_feats(state["backbone_out"], img_ids)
        self._tokens, self._mask = self.model.geometry_encoder(
            geo_prompt=prompt, img_feats=feats, img_sizes=sizes, img_pos_embeds=pos,
        )

    @torch.inference_mode()
    def add_exemplar(self, image, bbox: List[float], label: int = 1):
        """Add another exemplar (concatenates with existing)."""
        if not self.has_exemplar:
            return self.set_reference(image, bbox, label)
        old_t, old_m = self._tokens, self._mask
        self.set_reference(image, bbox, label)
        self._tokens = torch.cat([old_t, self._tokens], dim=0)
        self._mask = torch.cat([old_m, self._mask], dim=1)

    @torch.inference_mode()
    def segment(self, target_image, threshold: float = None) -> SegmentationResult:
        """Segment target image using cached exemplar tokens."""
        if not self.has_exemplar:
            raise RuntimeError("Call set_reference() first")

        threshold = threshold or self.threshold

        # Use processor for preprocessing + backbone
        state = self.processor.set_image(target_image)
        backbone_out = state["backbone_out"]
        orig_h, orig_w = state["original_height"], state["original_width"]

        # Add dummy text
        txt = self.model.backbone.forward_text(["visual"], device=self.device)
        backbone_out.update(txt)
        txt_feats = backbone_out["language_features"][:, self.processor.find_stage.text_ids]
        txt_mask = backbone_out["language_mask"][self.processor.find_stage.text_ids]

        # Build prompt: text + cached exemplar
        prompt = torch.cat([txt_feats, self._tokens], dim=0)
        prompt_mask = torch.cat([txt_mask, self._mask], dim=1)

        # Get target features
        img_ids = self.processor.find_stage.img_ids
        backbone_out, feats, pos, sizes = self.model._get_img_feats(backbone_out, img_ids)

        # Encoder + Decoder
        enc_out = self.model.transformer.encoder(
            src=feats, src_key_padding_mask=None, src_pos=pos,
            prompt=prompt, prompt_pos=torch.zeros_like(prompt),
            prompt_key_padding_mask=prompt_mask, feat_sizes=sizes,
        )

        query = self.model.transformer.decoder.query_embed.weight.unsqueeze(1)
        hs, ref_boxes, presence, _ = self.model.transformer.decoder(
            tgt=query, memory=enc_out["memory"],
            memory_key_padding_mask=enc_out.get("padding_mask"),
            pos=enc_out["pos_embed"], reference_boxes=None,
            level_start_index=enc_out["level_start_index"],
            spatial_shapes=enc_out["spatial_shapes"],
            valid_ratios=enc_out["valid_ratios"],
            tgt_mask=None, memory_text=prompt,
            text_attention_mask=prompt_mask, apply_dac=False,
        )

        hs = hs.transpose(1, 2)
        ref_boxes = ref_boxes.transpose(1, 2)

        # Predictions
        if self.model.use_dot_prod_scoring:
            logits = self.model.dot_prod_scoring(hs, prompt, prompt_mask)
        else:
            logits = self.model.class_embed(hs)

        from sam3.model.model_misc import inverse_sigmoid
        offsets = self.model.transformer.decoder.bbox_embed(hs)
        out_boxes = (inverse_sigmoid(ref_boxes) + offsets).sigmoid()

        # Scores with presence
        probs = logits[-1].sigmoid()
        if presence is not None:
            pres_val = presence[-1].sigmoid().mean().item()
            probs = probs * presence[-1].sigmoid().unsqueeze(1)
        else:
            pres_val = probs.max().item()

        probs = probs.squeeze(-1)[0]
        out_boxes = out_boxes[-1][0]

        # Masks
        masks = None
        if self.model.segmentation_head is not None:
            seg = self.model.segmentation_head(
                backbone_feats=backbone_out["backbone_fpn"], obj_queries=hs,
                image_ids=torch.zeros(1, dtype=torch.long, device=self.device),
                encoder_hidden_states=enc_out["memory"],
                prompt=prompt, prompt_mask=prompt_mask,
            )
            masks = seg.get("pred_masks")

        # Filter by threshold
        keep = probs > threshold
        probs, out_boxes = probs[keep], out_boxes[keep]
        if masks is not None:
            masks = masks[0, keep]

        # Postprocess boxes (same as Sam3Processor)
        boxes = box_ops.box_cxcywh_to_xyxy(out_boxes)
        scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], device=self.device)
        boxes = boxes * scale

        # Postprocess masks (same as Sam3Processor)
        if masks is not None and masks.numel() > 0:
            masks = interpolate(masks.unsqueeze(1), (orig_h, orig_w), mode="bilinear", align_corners=False)
            masks = (masks.sigmoid().squeeze(1) > 0.5).float()
        else:
            masks = torch.zeros(0, orig_h, orig_w, device=self.device)

        return SegmentationResult(masks=masks, boxes=boxes, scores=probs, presence_score=pres_val)

    def clear(self):
        """Clear cached exemplar tokens."""
        self._tokens = self._mask = None
