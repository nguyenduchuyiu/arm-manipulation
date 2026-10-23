"""One learned task token with the original pretrained MobileSAM mask decoder."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from mobile_sam import sam_model_registry
from mobile_sam.utils.transforms import ResizeLongestSide

WEIGHTS=Path('/workspace/robotics_assets/external/MobileSAM/weights/mobile_sam.pt')


class TaskMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.sam=sam_model_registry['vit_t'](checkpoint=str(WEIGHTS))
        self.sam.requires_grad_(False)
        self.sam.mask_decoder.requires_grad_(True)
        # Only the single-mask branch is supervised; multimask readout MLPs are unused.
        for head in self.sam.mask_decoder.output_hypernetworks_mlps[1:]:head.requires_grad_(False)
        self.task_prompt=nn.Embedding(4,256)
        with torch.no_grad():
            self.task_prompt.weight.copy_(self.sam.prompt_encoder.point_embeddings[1].weight+
                                          torch.randn_like(self.task_prompt.weight)*.02)
        self.resize=ResizeLongestSide(1024)

    def train(self,mode=True):
        super().train(mode)
        self.sam.image_encoder.eval();self.sam.prompt_encoder.eval()
        return self

    @torch.inference_mode()
    def encode(self,rgb):
        image=self.resize.apply_image(rgb)
        pixels=torch.as_tensor(image.copy()).permute(2,0,1)[None].to(self.task_prompt.weight.device)
        return self.sam.image_encoder(self.sam.preprocess(pixels))

    def forward(self,embeddings,target):
        assert embeddings.shape[1:]==(256,64,64) and target.shape==(len(embeddings),)
        d=self.sam.mask_decoder;b=len(target)
        output=torch.cat((d.iou_token.weight,d.mask_tokens.weight),dim=0)[None].expand(b,-1,-1)
        tokens=torch.cat((output,self.task_prompt(target)[:,None]),dim=1)
        dense=self.sam.prompt_encoder.no_mask_embed.weight.reshape(1,256,1,1)
        pe=self.sam.prompt_encoder.get_dense_pe().expand(b,-1,-1,-1)
        # Original decoder repeats one image for many prompts; here images/prompts are paired.
        hidden,src=d.transformer(embeddings+dense,pe,tokens)
        src=src.transpose(1,2).reshape(b,256,64,64)
        up=d.output_upscaling(src)
        hyper=d.output_hypernetworks_mlps[0](hidden[:,1])
        low=(hyper[:,None]@up.flatten(2)).reshape(b,1,256,256)
        logits=self.sam.postprocess_masks(low,(768,1024),(240,320))[:,0]
        quality=d.iou_prediction_head(hidden[:,0])[:,0]
        return logits,quality

    def train_state(self):
        return dict(task_prompt=self.task_prompt.state_dict(),mask_decoder=self.sam.mask_decoder.state_dict())

    def load_train_state(self,state):
        self.task_prompt.load_state_dict(state['task_prompt'],strict=True)
        self.sam.mask_decoder.load_state_dict(state['mask_decoder'],strict=True)


def loss_parts(logits,truth,quality):
    truth=truth.float();prob=logits.sigmoid()
    bce=F.binary_cross_entropy_with_logits(logits,truth)
    dice=(1-(2*(prob*truth).sum((1,2))+1)/(prob.sum((1,2))+truth.sum((1,2))+1)).mean()
    with torch.no_grad():
        mask=logits>0;gt=truth.bool()
        intersection=(mask & gt).sum((1,2)).float();union=(mask|gt).sum((1,2)).float()
        iou=(intersection+1e-6)/(union+1e-6)
    calibration=F.mse_loss(quality,iou)
    return bce+dice+.1*calibration,dict(bce=bce,dice=dice,quality=calibration,iou=iou.mean())


def overlay(rgb,truth,prediction):
    image=rgb.astype(np.float32).copy()
    image[truth]=image[truth]*.55+np.array([255,145,35])*.45
    image[prediction]=image[prediction]*.55+np.array([0,210,220])*.45
    return Image.fromarray(np.clip(image,0,255).astype(np.uint8))
