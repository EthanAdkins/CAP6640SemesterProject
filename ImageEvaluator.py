# References:
# https://huggingface.co/docs/diffusers/en/conceptual/evaluation

import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial

class Clip_Score_Eval:
    def __init__(self):
        self.clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    def calculate_clip_score(self, images, prompts):
        images_int = (images * 255).astype("uint8")
        clip_score = self.clip_score_fn(torch.from_numpy(images_int).permute(0,3,1,2), prompts).detach()
        return round(float(clip_score), 4)