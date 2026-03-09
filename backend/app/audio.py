import librosa
import numpy as np
import torch

vggish = torch.hub.load(
    "harritaylor/torchvggish",
    "vggish",
    pretrained=True
)
vggish.eval()


def extract_hlla(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    return torch.tensor(
        np.concatenate([mfcc.mean(axis=1), [pitch.mean()]]),
        dtype=torch.float32
    )


def extract_dhla(audio_path):
    with torch.no_grad():
        emb = vggish(audio_path)
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)
    return emb.mean(dim=0)


def combine_audio_features(hlla, dhla):
    return torch.cat([hlla, dhla]).unsqueeze(0)
