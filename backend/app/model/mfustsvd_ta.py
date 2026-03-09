import torch
import torch.nn as nn


# --------------------------------------------------
# Unimodal Encoder
# --------------------------------------------------
class UnimodalEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        weights = torch.softmax(x, dim=-1)
        return torch.sum(weights * x, dim=-1)


# --------------------------------------------------
# Self-Modality Transformer (B-SMTLMF)
# --------------------------------------------------
class SelfModalityTransformer(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


# --------------------------------------------------
# Cross-Modality Transformer (B-CMTLRMF)
# --------------------------------------------------
class CrossModalityTransformer(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key, value):
        attn_out, _ = self.attn(query, key, value)
        return self.norm(query + attn_out)


# --------------------------------------------------
# Low-Rank Fusion (Text × Audio)
# --------------------------------------------------
class LowRankFusionTA(nn.Module):
    def __init__(self, d_text, d_audio, d_out, rank=4):
        super().__init__()
        self.rank = rank
        self.Wt = nn.Parameter(torch.randn(rank, d_text))
        self.Wa = nn.Parameter(torch.randn(rank, d_audio))
        self.Wo = nn.Parameter(torch.randn(rank, d_out))

    def forward(self, Ft, Fa):
        fusion = 0
        for r in range(self.rank):
            tr = Ft @ self.Wt[r]
            ar = Fa @ self.Wa[r]
            fusion += (tr * ar).unsqueeze(1) * self.Wo[r]
        return fusion


# --------------------------------------------------
# Classifier
# --------------------------------------------------
class MFusTSVDClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)


# --------------------------------------------------
# MFusTSVD-TA (MAIN MODEL)
# --------------------------------------------------
class MFusTSVD_TA(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()

        self.text_encoder = UnimodalEncoder(768, d_model)
        self.audio_encoder = UnimodalEncoder(169, d_model)

        self.self_text = SelfModalityTransformer(d_model)
        self.self_audio = SelfModalityTransformer(d_model)

        self.cross = CrossModalityTransformer(d_model)

        self.lrmf = LowRankFusionTA(d_model, d_model, d_model)
        self.classifier = MFusTSVDClassifier(d_model)

    def forward(self, text_feat, audio_feat):
        # Encode
        text_enc = self.text_encoder(text_feat.transpose(1, 2)).unsqueeze(1)
        audio_enc = self.audio_encoder(audio_feat.unsqueeze(-1)).unsqueeze(1)

        # Self-modality
        text_self = self.self_text(text_enc)
        audio_self = self.self_audio(audio_enc)

        # Cross-modality
        text_cross = self.cross(text_self, audio_self, audio_self)
        audio_cross = self.cross(audio_self, text_self, text_self)

        # Fusion
        fusion = self.lrmf(
            text_cross.mean(1),
            audio_cross.mean(1)
        )

        return self.classifier(fusion)
