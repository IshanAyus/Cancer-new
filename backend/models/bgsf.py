#!/usr/bin/env python3
"""BGSF (Boundary-Guided Segmentation Fusion) Model

Matches the architecture from your paper.
"""
import timm
import torch
import torch.nn.functional as F
from torch import nn


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BGSFFusion(nn.Module):
    """Direction-aware fusion module from your paper"""
    def __init__(self, alpha_init=0.3):
        super().__init__()
        self.alpha_raw = nn.Parameter(torch.tensor(alpha_init))
        self.refine = nn.Sequential(
            ConvBNAct(3, 32, 3),
            nn.Conv2d(32, 1, 1)
        )
    
    def forward(self, l_seg, l_bnd):
        p_seg = torch.sigmoid(l_seg)
        p_bnd = torch.sigmoid(l_bnd)
        
        # Uncertainty map U = 4 * p_seg * (1 - p_seg)
        U = 4 * p_seg * (1 - p_seg)
        
        # Learnable alpha in (0, 1)
        alpha = torch.sigmoid(self.alpha_raw)
        
        # Direction-aware sharpening: p_tilde = p_seg + alpha * p_bnd * U * (p_seg - 0.5)
        p_tilde = p_seg + alpha * p_bnd * U * (p_seg - 0.5)
        p_tilde = torch.clamp(p_tilde, 0, 1)
        
        # Refinement network
        delta = self.refine(torch.cat([p_tilde, p_seg, p_bnd], dim=1))
        l_fused = l_seg + delta
        
        return l_fused, p_tilde


class SegDecoder(nn.Module):
    def __init__(self, in_channels_list, out_ch=256):
        super().__init__()
        self.projections = nn.ModuleList([
            ConvBNAct(c, out_ch, 1) for c in in_channels_list
        ])
        self.fuse = nn.Sequential(
            ConvBNAct(out_ch * 4, out_ch, 3),
            ConvBNAct(out_ch, out_ch, 3),
            nn.Conv2d(out_ch, 1, 1)
        )
    
    def forward(self, features, target_size):
        projected = []
        for i, (feat, proj) in enumerate(zip(features, self.projections)):
            p = proj(feat)
            p = F.interpolate(p, size=target_size, mode='bilinear', align_corners=False)
            projected.append(p)
        concat = torch.cat(projected, dim=1)
        return self.fuse(concat)


class BGSF(nn.Module):
    def __init__(self, encoder_name='swin_tiny_patch4_window7_224', pretrained=True):
        super().__init__()
        
        # Swin-Tiny encoder
        self.encoder = timm.create_model(
            encoder_name, 
            pretrained=pretrained, 
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        encoder_channels = self.encoder.feature_info.channels()  # [96, 192, 384, 768]
        
        # Decoders at 1/4 resolution (H/4, W/4)
        _target_w = 256 // 4  # 64
        
        self.seg_decoder = SegDecoder(encoder_channels, 256)
        self.bnd_decoder = SegDecoder(encoder_channels, 128)  # half capacity
        
        self.fusion = BGSFFusion()
        
        # Final upsample to full resolution
        self.final_upsample = lambda x: F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
    
    def forward(self, x):
        # x: [B, 3, H, W] where H=W=256
        _B, _, H, W = x.shape
        h4, w4 = H // 4, W // 4
        
        # Encoder
        feats = self.encoder(x)  # List of 4 feature maps
        
        # Segmentation decoder
        l_seg_4x = self.seg_decoder(feats, (h4, w4))
        l_seg = self.final_upsample(l_seg_4x)
        
        # Boundary decoder
        l_bnd_4x = self.bnd_decoder(feats, (h4, w4))
        l_bnd = self.final_upsample(l_bnd_4x)
        
        # Fusion
        l_fused, p_tilde = self.fusion(l_seg, l_bnd)
        
        return l_seg, l_bnd, l_fused, p_tilde


def load_bgsf(checkpoint_path, device='cpu'):
    """Load BGSF from checkpoint"""
    model = BGSF(pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle 'backbone.' prefix for encoder weights
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_state_dict[k[9:]] = v  # strip 'backbone.'
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device).eval()
    return model


if __name__ == "__main__":
    # Test model
    model = BGSF(pretrained=False)
    x = torch.randn(1, 3, 256, 256)
    l_seg, l_bnd, l_fused, p_tilde = model(x)
    print(f"l_seg: {l_seg.shape}")
    print(f"l_bnd: {l_bnd.shape}")
    print(f"l_fused: {l_fused.shape}")
    print(f"p_tilde: {p_tilde.shape}")
    print("Model test passed!")