import librosa
import numpy as np
import torch

# --------------------------------------------------
# DEVICE CONFIGURATION
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Audio features running on:", device)

# --------------------------------------------------
# LOAD VGGISH MODEL (ONCE)
# --------------------------------------------------

vggish = torch.hub.load(
    "harritaylor/torchvggish",
    "vggish",
    pretrained=True
)

vggish.eval()

# Move to GPU
vggish = vggish.to(device)

# Disable gradients (important for speed)
for param in vggish.parameters():
    param.requires_grad = False


# --------------------------------------------------
# HIGH-LEVEL LOW-LEVEL AUDIO FEATURES (HLLA)
# --------------------------------------------------

def extract_hlla(audio_path):

    # Load audio
    y, sr = librosa.load(
        audio_path,
        sr=16000
    )

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    # Pitch
    pitch = librosa.yin(
        y,
        fmin=50,
        fmax=300
    )

    features = np.concatenate([
        mfcc.mean(axis=1),
        [pitch.mean()]
    ])

    return torch.tensor(
        features,
        dtype=torch.float32
    )


# --------------------------------------------------
# DEEP HIGH-LEVEL AUDIO FEATURES (DHLA)
# --------------------------------------------------

def extract_dhla(audio_path):

    with torch.no_grad():

        emb = vggish(audio_path)

    # Ensure correct shape

    if emb.dim() == 1:
        emb = emb.unsqueeze(0)

    dhla = emb.mean(dim=0)

    return dhla.cpu()


# --------------------------------------------------
# COMBINE FEATURES
# --------------------------------------------------

def combine_audio_features(hlla, dhla):

    # Ensure tensors

    if not isinstance(hlla, torch.Tensor):
        hlla = torch.tensor(hlla)

    if not isinstance(dhla, torch.Tensor):
        dhla = torch.tensor(dhla)

    combined = torch.cat(
        [hlla, dhla],
        dim=0
    )

    # Add batch dimension

    return combined.unsqueeze(0)