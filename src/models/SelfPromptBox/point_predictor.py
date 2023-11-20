from torch import nn
import torch

from src.models.SelfPromptBox.transformer import TransformerDecoder,TransformerDecoderLayer
from torch.nn import functional as F


class PointDetector(nn.Module):
    def __init__(self,
        hidden_dim=256,
        nhead=8,
        num_classes=1,
        dim_feedforward=2048,
        num_queries=100,
        ) -> None:
        self,
        super().__init__()
        self.num_queries=num_queries
        self.hidden_dim=hidden_dim
        self.num_classes=1
        self.dim_feedforward=dim_feedforward
        self.nhead=nhead
        self.num_classes=num_classes
        self.point_detector_layer=TransformerDecoderLayer(d_model=self.hidden_dim,
                                                       nhead=self.nhead,
                                                       dim_feedforward=self.dim_feedforward,
                                                       dropout=0.1,
                                                       activation='relu',
                                                       normalize_before=False)
        self.point_detector = TransformerDecoder(self.point_detector_layer, num_layers=6,
                                             norm=nn.LayerNorm(self.hidden_dim),
                                             return_intermediate=True)
        self.query_embed=nn.Embedding(self.num_queries,self.hidden_dim)
        self.class_embed = nn.Linear(self.hidden_dim, self.num_classes + 1)
        self.point_embed = MLP(self.hidden_dim, self.hidden_dim, 2, 3)
        self.query_embed=nn.Embedding(self.num_queries, self.hidden_dim)
    def forward(self,
        image_embeddings: torch.Tensor,
        pos:torch.Tensor):

        bs,c,w,h=image_embeddings.shape
        query_embed= self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        memory=image_embeddings.flatten(2).permute(2,0,1) # (n_tokens,bs,hidden_dim)
        hs=self.point_detector(tgt,memory,
                            memory_key_padding_mask=None,
                            pos=pos,
                            query_pos=query_embed)
                            
        hs=hs.transpose(1,2) # n_decoders,bs,n_query,hidden_dim
        memory=memory.permute(1,2,0).view(bs,c,h,w)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.point_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x