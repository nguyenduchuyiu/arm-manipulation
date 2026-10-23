"""Official frozen SAM3 BF16 text masks; no GT or task-specific learned prompts."""
import time

import numpy as np
from PIL import Image
import torch

from OATFlow.task_mask.sam3_evaluate import PROMPTS,REVISION,digest


class Sam3Masks:
    def __init__(self):
        from huggingface_hub import snapshot_download
        from transformers import Sam3Model,Sam3Processor
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        assert torch.cuda.is_bf16_supported()
        path=snapshot_download(repo_id='facebook/sam3',revision=REVISION,token=True,
            cache_dir='/workspace/robotics_assets/pretrained/sam3_hf',allow_patterns=['*.json','*.safetensors','*.txt'])
        self.model=Sam3Model.from_pretrained(path,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
        self.processor=Sam3Processor.from_pretrained(path);self.output_class=BaseModelOutputWithPooling
        assert {p.dtype for p in self.model.parameters() if p.is_floating_point()}=={torch.bfloat16}
        self.text=[];self.text_masks=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for prompt in PROMPTS:
                tokens=self.processor(text=[prompt]*2,return_tensors='pt').to('cuda')
                self.text.append(self.model.get_text_features(**tokens).pooler_output)
                self.text_masks.append(tokens['attention_mask'])
        from pathlib import Path
        self.weights=Path(path)/'model.safetensors';self.weights_sha256=digest(self.weights)
        self.calls=0;self.seconds=0.
        self.verification=None

    @torch.inference_mode()
    def __call__(self,images,targets):
        assert len(images)==len(targets)==2 and targets[0]==targets[1]
        target=int(targets[0]);started=time.perf_counter()
        inputs=self.processor(images=[Image.fromarray(rgb) for rgb in images],return_tensors='pt')
        pixels=inputs['pixel_values'].to(device='cuda',dtype=torch.bfloat16)
        text=self.output_class(pooler_output=self.text[target])
        attention=self.text_masks[target]
        with torch.autocast('cuda',dtype=torch.bfloat16):output=self.model(pixel_values=pixels,text_embeds=text,attention_mask=attention)
        if self.verification is None:
            tokens=self.processor(text=[PROMPTS[target]]*2,return_tensors='pt').to('cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):reference=self.model(pixel_values=pixels,**tokens)
            delta=float((output.pred_masks.float()-reference.pred_masks.float()).abs().max())
            assert delta<.02,f'Cached-text forward mismatch: {delta}'
            self.verification=dict(status='passed',two_view_batch=2,cached_text_max_logit_difference=delta,
                parameter_dtype='torch.bfloat16',mask_logits_dtype=str(output.pred_masks.dtype),GT_used_as_input=False)
            del reference
        results=self.processor.post_process_instance_segmentation(output,threshold=.5,mask_threshold=.5,target_sizes=[(240,320),(240,320)])
        masks=[];metadata=[]
        for result in results:
            scores=result['scores'].float().cpu().numpy();best=int(scores.argmax()) if len(scores) else -1
            mask=result['masks'][best].cpu().numpy().astype(bool) if best>=0 else np.zeros((240,320),bool)
            assert mask.shape==(240,320)
            masks.append(mask);metadata.append(dict(prompt=PROMPTS[target],detections=len(scores),selected_index=best,
                confidence=float(scores[best]) if best>=0 else None,mask_pixels=int(mask.sum())))
        self.calls+=1;self.seconds+=time.perf_counter()-started
        return masks,metadata
