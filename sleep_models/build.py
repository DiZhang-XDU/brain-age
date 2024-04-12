from .swin_transformer import SwinTransformer, PatchMerging
from torch import nn


class Build_Merging(nn.Module):
    def __init__(self, body_psg, body_hyp, head) -> None:
        super().__init__()
        self.body_eeg = body_psg
        self.body_hyp = body_hyp
        self.head = head
    def forward(self, x_eeg, x_hyp, t, alpha=.5):
        x1, _ = self.body_eeg(x_eeg)
        x2, _ = self.body_hyp(x_hyp)
        y, feat= self.head(x1, x2, t)
        return y, feat

class _body_Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x, *args):
        return x, x

def build_model(cfg):
    # PSG Feat Extract  
    if cfg.EEG_NET == 'swin':
        body_psg = SwinTransformer(image_size=cfg.SWIN.IN_LEN,
                                    patch_size=cfg.SWIN.PATCH_SIZE,
                                    in_chans=cfg.SWIN.IN_CHANS,
                                    num_classes=cfg.SWIN.OUT_CHANS,
                                    embed_dim=cfg.SWIN.EMBED_DIM,
                                    depths=cfg.SWIN.DEPTHS,
                                    num_heads=cfg.SWIN.NUM_HEADS,
                                    window_size=cfg.SWIN.WINDOW_SIZE,
                                    mlp_ratio=cfg.SWIN.MLP_RATIO,
                                    qkv_bias=cfg.SWIN.QKV_BIAS,
                                    qk_scale=cfg.SWIN.QK_SCALE,
                                    drop_rate=cfg.SWIN.DROP_RATE,
                                    drop_path_rate=cfg.SWIN.DROP_PATH_RATE,
                                    ape=cfg.SWIN.APE,
                                    patch_norm=cfg.SWIN.PATCH_NORM,
                                    head=cfg.HEAD,
                                    use_checkpoint=cfg.USE_CHECKPOINT)
    elif cfg.EEG_NET == 'abrinkk':
        from sleep_models.comparison_abrinkk import M_PSG2FEAT
        body_psg = M_PSG2FEAT()
    elif cfg.EEG_NET == 'tiny':
        from sleep_models.comparison_tiny import TinySleepNet
        body_psg = TinySleepNet()
    elif cfg.EEG_NET == 'resatt':
        from sleep_models.comparison_resatt import Stage_Net_E2E
        body_psg = Stage_Net_E2E()
    elif cfg.EEG_NET == 'tfm':
        from sleep_models.comparison_tfm import Epoch_Cross_Transformer_Network
        body_psg = Epoch_Cross_Transformer_Network()
    else:
        body_psg = _body_Identity()

    # Hypnogram Feat Extract
    if cfg.HYP_NET == 'cnn':
        from .hypResnet import HypResnet
        body_hyp = HypResnet()
    else:
        body_hyp = _body_Identity()

    # Merging Head
    from .heads import HeadMergning, HeadEEG, HeadHYP
    if cfg.HEAD == 'merge':
        head = HeadMergning(cfg)
    elif cfg.HEAD == 'eeg_only':
        head = HeadEEG(cfg)
    elif cfg.HEAD == 'hyp_only':
        head = HeadHYP(cfg)
    else:
        head = _body_Identity()
        
    model = Build_Merging(body_psg, body_hyp, head)

    return model
