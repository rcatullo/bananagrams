"""
Image Editing Mask Prediction Model.

Architecture:
1. Image Encoder: ResNet-50 with coordinate channels
2. Text Encoder: CLIP text tower
3. Cross-Attention Fusion: Image features attend to text features
4. Mask Decoder: U-Net style decoder with skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import clip
from typing import Dict, List, Tuple


class CoordinateChannels(nn.Module):
    """Add 2D coordinate channels to spatial features."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map
        
        Returns:
            (B, C+2, H, W) feature map with coordinate channels
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Create normalized coordinate grids [0, 1]
        y_coords = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        
        # Concatenate coordinate channels
        return torch.cat([x, y_coords, x_coords], dim=1)


class ImageEncoder(nn.Module):
    """
    ResNet-based image encoder with multi-scale features.
    
    Extracts features at multiple scales for U-Net skip connections.
    """
    
    def __init__(self, backbone='resnet50', pretrained=True, freeze=False):
        """
        Args:
            backbone: ResNet variant ('resnet50', 'resnet101')
            pretrained: Whether to use ImageNet pretrained weights
            freeze: Whether to freeze backbone weights
        """
        super().__init__()
        
        # Load ResNet backbone
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # C2: 256 channels
        self.layer2 = resnet.layer2  # C3: 512 channels
        self.layer3 = resnet.layer3  # C4: 1024 channels
        self.layer4 = resnet.layer4  # C5: 2048 channels
        
        self.stage_channels = {
            'C1': 64,   # After conv1
            'C2': 256,  # After layer1
            'C3': 512,  # After layer2
            'C4': 1024, # After layer3
            'C5': 2048  # After layer4
        }
        
        self.coord_channels = CoordinateChannels()
        
        if freeze:
            self.freeze()
    
    def freeze(self):
        """Freeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input image
        
        Returns:
            Dictionary with multi-scale features:
            - 'features': Main feature map with coordinate channels
            - 'C1', 'C2', 'C3', 'C4': Skip connection features
        """
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1_pool = self.maxpool(c1)
        
        # ResNet stages
        c2 = self.layer1(c1_pool)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # Add coordinate channels to main features
        c5_with_coords = self.coord_channels(c5)
        
        return {
            'features': c5_with_coords,  # (B, 2048+2, H/32, W/32)
            'C1': c1,     # (B, 64, H/2, W/2)
            'C2': c2,     # (B, 256, H/4, W/4)
            'C3': c3,     # (B, 512, H/8, W/8)
            'C4': c4,     # (B, 1024, H/16, W/16)
        }


class TextEncoder(nn.Module):
    """CLIP-based text encoder."""
    
    def __init__(self, model_name='ViT-B/32', freeze=False):
        """
        Args:
            model_name: CLIP model variant
            freeze: Whether to freeze text encoder weights
        """
        super().__init__()
        
        # Load CLIP model
        self.clip_model, _ = clip.load(model_name, device='cpu')
        
        self.text_encoder = self.clip_model
        
        self.embed_dim = self.clip_model.text_projection.shape[1]
        
        if freeze:
            self.freeze()
    
    def freeze(self):
        """Freeze all text encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all text encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(self, text_tokens):
        """
        Args:
            text_tokens: (B, seq_len) tokenized text
        
        Returns:
            (B, seq_len, embed_dim) text features
        """
        x = self.text_encoder.token_embedding(text_tokens)  # (B, seq_len, dim)
        x = x + self.text_encoder.positional_embedding
        x = x.permute(1, 0, 2)  # (seq_len, B, dim)
        x = self.text_encoder.transformer(x)
        x = x.permute(1, 0, 2)  # (B, seq_len, dim)
        x = self.text_encoder.ln_final(x)
        
        return x


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer for image-text fusion."""
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        """
        Args:
            hidden_dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, image_features, text_features):
        """
        Args:
            image_features: (B, N_img, hidden_dim) image tokens
            text_features: (B, N_text, hidden_dim) text tokens
        
        Returns:
            (B, N_img, hidden_dim) fused features
        """
        # Cross-attention: image attends to text
        attn_output, _ = self.multihead_attn(
            query=image_features,
            key=text_features,
            value=text_features
        )
        
        # Residual + normalization
        image_features = self.norm1(image_features + attn_output)
        
        ffn_output = self.ffn(image_features)
        image_features = self.norm2(image_features + ffn_output)
        
        return image_features


class CrossAttentionFusion(nn.Module):
    """Multi-layer cross-attention for image-text fusion."""
    
    def __init__(self, image_dim, text_dim, hidden_dim, num_layers=4, num_heads=8, dropout=0.1):
        """
        Args:
            image_dim: Input image feature dimension
            text_dim: Input text feature dimension
            hidden_dim: Hidden dimension for attention
            num_layers: Number of cross-attention layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Project image and text features to common dimension
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, image_features, text_features, spatial_shape):
        """
        Args:
            image_features: (B, C_img, H, W) spatial image features
            text_features: (B, N_text, C_text) text features
            spatial_shape: (H, W) to reshape output
        
        Returns:
            (B, hidden_dim, H, W) fused spatial features
        """
        B, C_img, H, W = image_features.shape
        
        # Flatten spatial dimensions: (B, C_img, H, W) -> (B, H*W, C_img)
        image_flat = image_features.flatten(2).permute(0, 2, 1)
        
        # Project to common dimension
        image_tokens = self.image_proj(image_flat)  # (B, H*W, hidden_dim)
        text_tokens = self.text_proj(text_features)  # (B, N_text, hidden_dim)
        
        # Apply cross-attention layers
        fused_tokens = image_tokens
        for layer in self.layers:
            fused_tokens = layer(fused_tokens, text_tokens)
        
        fused_tokens = self.output_proj(fused_tokens)
        
        # Reshape back to spatial: (B, H*W, hidden_dim) -> (B, hidden_dim, H, W)
        fused_spatial = fused_tokens.permute(0, 2, 1).reshape(B, self.hidden_dim, H, W)
        
        return fused_spatial


class DecoderBlock(nn.Module):
    """Single decoder block with upsampling and skip connections."""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        """
        Args:
            in_channels: Input channels from previous layer
            skip_channels: Channels from skip connection
            out_channels: Output channels
        """
        super().__init__()
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, 
            kernel_size=2, stride=2
        )
        
        # Convolution after concatenation
        concat_channels = in_channels // 2 + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        """
        Args:
            x: (B, in_channels, H, W) input from previous layer
            skip: (B, skip_channels, 2H, 2W) skip connection
        
        Returns:
            (B, out_channels, 2H, 2W) upsampled and fused features
        """
        # Upsample
        x = self.upsample(x)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Convolution
        x = self.conv(x)
        
        return x


class MaskDecoder(nn.Module):
    """U-Net style decoder for mask prediction."""
    
    def __init__(self, input_channels, skip_channels_list, base_channels=64):
        """
        Args:
            input_channels: Channels from fusion module
            skip_channels_list: List of channels for skip connections [C4, C3, C2, C1]
            base_channels: Base number of channels for decoder
        """
        super().__init__()
        
        # Initial projection
        self.input_proj = nn.Conv2d(input_channels, base_channels * 16, kernel_size=1)
        
        # Decoder blocks (progressively upsample)
        self.decoder4 = DecoderBlock(base_channels * 16, skip_channels_list[0], base_channels * 8)
        self.decoder3 = DecoderBlock(base_channels * 8, skip_channels_list[1], base_channels * 4)
        self.decoder2 = DecoderBlock(base_channels * 4, skip_channels_list[2], base_channels * 2)
        self.decoder1 = DecoderBlock(base_channels * 2, skip_channels_list[3], base_channels)
        
        # Final upsampling to original resolution
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Output head (1x1 conv to single channel mask)
        self.output_head = nn.Conv2d(base_channels, 1, kernel_size=1)
    
    def forward(self, x, skip_connections):
        """
        Args:
            x: (B, input_channels, H, W) fused features
            skip_connections: Dict with 'C4', 'C3', 'C2', 'C1' skip features
        
        Returns:
            (B, 1, H_out, W_out) mask logits
        """
        # Initial projection
        x = self.input_proj(x)
        
        # Progressive upsampling with skip connections
        x = self.decoder4(x, skip_connections['C4'])  # H/16, W/16
        x = self.decoder3(x, skip_connections['C3'])  # H/8, W/8
        x = self.decoder2(x, skip_connections['C2'])  # H/4, W/4
        x = self.decoder1(x, skip_connections['C1'])  # H/2, W/2
        
        # Final upsampling to original resolution
        x = self.final_upsample(x)  # H, W
        
        # Output head
        mask_logits = self.output_head(x)
        
        return mask_logits


class MaskPredictionModel(nn.Module):
    """Complete model for image editing mask prediction."""
    
    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        
        # Image encoder
        self.image_encoder = ImageEncoder(
            backbone=config['model']['image_encoder']['backbone'],
            pretrained=config['model']['image_encoder']['pretrained'],
            freeze=config['model']['image_encoder']['freeze_backbone']
        )
        
        # Text encoder
        self.text_encoder = TextEncoder(
            model_name=config['model']['text_encoder']['model_name'],
            freeze=config['model']['text_encoder']['freeze']
        )
        
        # Get dimensions
        image_feat_dim = self.image_encoder.stage_channels['C5'] + 2  # +2 for coordinate channels
        text_feat_dim = self.text_encoder.embed_dim
        hidden_dim = config['model']['fusion']['hidden_dim']
        
        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            image_dim=image_feat_dim,
            text_dim=text_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=config['model']['fusion']['num_layers'],
            num_heads=config['model']['fusion']['num_heads'],
            dropout=config['model']['fusion']['dropout']
        )
        
        # Mask decoder
        skip_channels = [
            self.image_encoder.stage_channels['C4'],  # 1024
            self.image_encoder.stage_channels['C3'],  # 512
            self.image_encoder.stage_channels['C2'],  # 256
            self.image_encoder.stage_channels['C1'],  # 64
        ]
        
        self.decoder = MaskDecoder(
            input_channels=hidden_dim,
            skip_channels_list=skip_channels,
            base_channels=config['model']['decoder']['base_channels']
        )
    
    def forward(self, image, text_tokens):
        """
        Args:
            image: (B, 3, H, W) input image
            text_tokens: (B, seq_len) tokenized text
        
        Returns:
            (B, 1, H, W) mask logits
        """
        # Encode image (multi-scale features)
        image_feats = self.image_encoder(image)
        spatial_features = image_feats['features']  # (B, 2048+2, H/32, W/32)
        
        # Encode text
        text_feats = self.text_encoder(text_tokens)  # (B, seq_len, text_dim)
        
        # Fusion via cross-attention
        B, C, H, W = spatial_features.shape
        fused_features = self.fusion(spatial_features, text_feats, (H, W))
        
        # Decode to mask
        mask_logits = self.decoder(fused_features, image_feats)
        
        return mask_logits


def build_model(config):
    """
    Build the model.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        MaskPredictionModel instance
    """
    model = MaskPredictionModel(config)
    return model

