import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_transformer import SwinTransformer,SwinTransformerBlock, BasicLayer, PatchMerging, PatchEmbed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from sleep_models.hypResnet import HypResnet

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchMerging_6x1(nn.Module):
    def __init__(self, input_len, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_len = input_len
        self.dim = dim
        self.norm = norm_layer(6 * dim)
        self.reduction = nn.Linear(6 * dim, dim, bias=False)
        
    def forward(self, x):
        """
        x: B, L, C
        """
        L = self.input_len
        B, Lx, C = x.shape
        assert L == Lx, "input feature has wrong size"
        assert L % 6 == 0, f"x size ({L}) is not a multiple of 6."
        x0 = x[:, 0::6, :]  # B L/6 C
        x1 = x[:, 1::6, :]  # B L/6 C
        x2 = x[:, 2::6, :]  # B L/6 C
        x3 = x[:, 3::6, :]  # B L/6 C
        x4 = x[:, 4::6, :]  # B L/6 C
        x5 = x[:, 5::6, :]  # B L/6 C
        x = torch.cat([x0, x1, x2, x3, x4, x5], -1)  # B L/6 6*C
        x = x.view(B, -1, 6 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        return x

class GAP_head(nn.Module):
    def __init__(self, dim = 48 ,kernel = 8) -> None:
        super().__init__()
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(dim)
        self.avgpool = nn.AvgPool1d(kernel_size=kernel, stride=kernel)
    def forward(self, x):
        """
        x: B, L, C
        """
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = x.transpose(1,2)
        return x

class BasicLayer_up(nn.Module):
    def __init__(self, dim, input_len, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_len = input_len
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_len=input_len ,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(input_len=input_len, dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class PatchExpand(nn.Module):
    def __init__(self, input_len, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_len = input_len
        self.dim = dim
        self.expand = nn.Linear(dim, 4*dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, L, C
        """
        B, Lx, C = x.shape
        L = self.input_len
        assert Lx == L, "input feature has wrong size"
        x = self.expand(x)
        
        x = x.view(B, -1, C)
        x= self.norm(x)
        return x

#### Heads

class HeadEEG(nn.Module):
    def __init__(self, cfg, ape = False) -> None:
        super().__init__()
        self.ape = ape
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if cfg.criterion == 'DecadeCE':
            self.fc_eeg = nn.Linear(cfg.SWIN.EMBED_DIM, cfg.SWIN.OUT_CHANS)
        else:
            self.fc_eeg = nn.Linear(cfg.SWIN.OUT_CHANS, 1, bias=True)
            self.fc_eeg.bias.data = torch.Tensor([50.0]) 

    def forward(self, x_eeg:torch.Tensor, x_hyp:torch.Tensor, t):   
        # inputs:
        # x_eeg [bs, L(Window_Size), dim(48)]
        B, _, C = x_eeg.shape
        x_eeg = self.avgpool(x_eeg.swapaxes(1,2)).reshape(B,-1)
        out_eeg = self.fc_eeg(x_eeg)
        return out_eeg, x_eeg.detach().clone()

class HeadHYP(nn.Module):
    def __init__(self, cfg, ape = False) -> None:
        super().__init__()
        self.fc_hyp = nn.Linear(10, cfg.SWIN.OUT_CHANS)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x_eeg:torch.Tensor, x_hyp:torch.Tensor, t):   
        # inputs:
        # x_hyp [bs, L(40), dim(10)]
        B, _, C = x_hyp.shape
        x_hyp = self.avgpool(x_hyp.swapaxes(1,2)).reshape(B,-1)
        x_hyp = self.fc_hyp(x_hyp)
        return x_hyp, x_hyp.detach().clone()

class HeadMergning(nn.Module):
    def __init__(self, cfg, ape = False) -> None:
        super().__init__()
        self.ape = ape
        self.fc_hyp = nn.Linear(10, cfg.SWIN.OUT_CHANS) 
        self.fc_eeg = nn.Linear(cfg.SWIN.EMBED_DIM, cfg.SWIN.OUT_CHANS)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.avgpool_hyp = nn.AdaptiveAvgPool1d(1)
        if cfg.criterion == 'DecadeCE':
            self.fc_out = nn.Linear(cfg.SWIN.OUT_CHANS, cfg.SWIN.OUT_CHANS) 
        else:
            self.fc_out = nn.Linear(cfg.SWIN.OUT_CHANS, 1, bias=True)
            self.fc_out.bias.data = torch.Tensor([50.0]) 

    def forward(self, x_eeg:torch.Tensor, x_hyp:torch.Tensor, t, alpha = 0.005):   
        # inputs:
        # x_eeg [bs, L(Window_Size), dim(48)]
        # x_hyp [bs, L(40), dim(10)]
        B, _, C = x_eeg.shape
        x_eeg = self.avgpool(x_eeg.swapaxes(1,2)).reshape(B,-1)
        x_hyp = self.avgpool_hyp(x_hyp.swapaxes(1,2)).reshape(B,-1)

        out_eeg = self.fc_eeg(x_eeg)
        out_hyp = self.fc_hyp(x_hyp)

        gate = torch.sigmoid(out_hyp) * alpha
        out_feat = out_eeg + torch.mul(gate, out_eeg) 
        output = self.fc_out(out_feat)

        return output, torch.cat([x_eeg, x_hyp],1).detach().clone()

class HeadPost(nn.Module):
    def __init__(self, cfg, in_chans = 10):
        super().__init__()
        self.embed_dim = cfg.SWIN.EMBED_DIM
        self.layer = SwinTransformer(image_size=cfg.HYP_EPOCH//5,
                                    patch_size=9,
                                    in_chans=cfg.SWIN.EMBED_DIM,
                                    embed_dim=cfg.SWIN.EMBED_DIM,
                                    depths=[2],
                                    num_heads=[4],
                                    window_size=4,
                                    mlp_ratio=cfg.SWIN.MLP_RATIO,
                                    qkv_bias=cfg.SWIN.QKV_BIAS,
                                    qk_scale=cfg.SWIN.QK_SCALE,
                                    drop_rate=cfg.SWIN.DROP_RATE,
                                    drop_path_rate=cfg.SWIN.DROP_PATH_RATE,
                                    ape=cfg.SWIN.APE,
                                    patch_norm=cfg.SWIN.PATCH_NORM)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cfg.SWIN.EMBED_DIM, cfg.SWIN.OUT_CHANS)
        if in_chans > self.embed_dim:
            self.fc_hyp = nn.Linear(in_chans - cfg.SWIN.EMBED_DIM, cfg.SWIN.OUT_CHANS)
            self.fc_out = nn.Linear(cfg.SWIN.OUT_CHANS, cfg.SWIN.OUT_CHANS) 

    def forward(self, x, alpha = .005):
        # [B, C, L]
        if x.size(1) == self.embed_dim:
            x, _ = self.layer(x)
            x = self.avgpool(x.swapaxes(1,2)).reshape(x.size(0),-1)
            x = self.fc(x)
        else:
            x_eeg = x[:,:self.embed_dim,:]
            x_eeg, _ = self.layer(x_eeg)
            x_eeg = self.avgpool(x_eeg.swapaxes(1,2)).reshape(x_eeg.size(0),-1)
            x_eeg = self.fc(x_eeg)
            x_hyp = x[:,self.embed_dim:,:]
            x_hyp = self.avgpool(x_hyp).reshape(x_hyp.size(0),-1)
            x_hyp = self.fc_hyp(x_hyp)
            gate = torch.sigmoid(x_hyp) * alpha
            x = x_eeg + torch.mul(gate, x_eeg)
            x = self.fc_out(x)
        return x
    