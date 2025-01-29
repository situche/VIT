import torch
import torch.nn as nn
from einops import rearrange

class VIT(nn.Module):

    def __init__(self, *, img_dim, path_dim, in_channels=3, num_classes=10, dim=512, blocks=6, 
                 heads=4, dim_linear_block=1024, dropout=0, transformer=None, classification=True):
        super().__init__()
        assert img_dim % path_dim == 0, f'patch size {path_dim} not divisible'
        self.p = path_dim
        self.classification = classification
        tokens = (img_dim // path_dim) ** 2
        self.token_dim = in_channels * (path_dim ** 2)  # 
        self.dim = dim

        self.project_patches = nn.Linear(self.token_dim, dim)

        self.emb_dropout = nn.Dropout(dropout)
        if self.classification:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.pos_emb1D = nn.Parameter(torch.randn(tokens + 1, dim))
            self.mlp_head = nn.Linear(dim, num_classes)
        else:
            self.pos_emb1D = nn.Parameter(torch.randn(tokens, dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim_linear_block, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=blocks)

    def expand_cls_to_batch(self, batch):
        return self.cls_token.expand([batch, -1, -1])
        
    def forward(self, img, mask=None):
        batch_size = img.shape[0]
        img_patches = rearrange(img, 'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)', patch_x=self.p, patch_y=self.p)
        img_patches = self.project_patches(img_patches)

        if self.classification:
            img_patches = torch.cat((self.expand_cls_to_batch(batch_size), img_patches), dim=1)
            
        patch_embeddings = self.emb_dropout(img_patches + self.pos_emb1D)

        y = self.transformer(patch_embeddings, src_key_padding_mask=mask)

        if self.classification:
            return self.mlp_head(y[:, 0, :])
        else:
            return y
