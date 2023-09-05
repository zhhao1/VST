# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
from fairseq.tasks import FairseqTask
from fairseq.models import (
    BaseFairseqModel,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2_asr import (
    Wav2VecEncoder,
    Wav2VecCtc,
    Wav2Vec2CtcConfig,
)


@register_model("wav2vec2_laser", dataclass=Wav2Vec2CtcConfig)
class Wav2VecLaser(Wav2VecCtc):
    def __init__(self, cfg: Wav2Vec2CtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__(cfg, w2v_encoder)
        self.num_updates = 0
        self.freeze_finetune_updates = cfg.freeze_finetune_updates

    @classmethod
    def build_model(cls, cfg: Wav2Vec2CtcConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2VecEncoder(cfg, 1024)
        return cls(cfg, w2v_encoder)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        #print(output.keys())
        #print(len(output['layer_results']))
        #print(output['layer_results'][0].size())

        x_out = output["encoder_out"] * 0.01 # default
        #x_out = output["encoder_out"] * 0.001
        #x_out = output["encoder_out"]
        out_pad_mask = output["padding_mask"]
        #embedding = output['layer_results'][0]
        #print(x_out.size()) torch.Size([429, 1, 1024])
        #print(out_pad_mask.size()) torch.Size([1, 429])
        #print(output['layer_results'][0].size()) torch.Size([429, 1, 1920])
        #return torch.mean(x_out, dim=0).squeeze(1)
        #value = self.remove_pad_and_mean(x_out, out_pad_mask)
        #value = self.remove_pad_and_mean(embedding, out_pad_mask)
        #return value
        
        #return torch.mean(output['layer_results'][0], dim=0).squeeze(1) # embedding output

        # Set padded outputs to -inf so they are not selected by max-pooling
        if out_pad_mask is not None and out_pad_mask.any():
            x_out = (
                x_out.float()
                .masked_fill_(out_pad_mask.T.unsqueeze(-1), float("-inf"))
                .type_as(x_out)
            )
        return x_out.max(dim=0)[0]
        
    def remove_pad_and_mean(self, input_tensor, pad_mask):
        """
        input_tensor: T*1*C
        pad_mask: 1*T
        """
        input_tensor = input_tensor.squeeze(1) # T*C
        pad_mask = ~pad_mask.squeeze(0) # T
        length = input_tensor.size(0)
        mean_value = torch.mean(input_tensor[pad_mask], dim=0) # 1*C
        return mean_value
