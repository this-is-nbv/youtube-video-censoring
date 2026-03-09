import numpy as np
import soundfile as sf
import torch

from mfustsvd_ta import MFusTSVD_TA

audio_path = "test.wav"
sr = 16000
sf.write(audio_path, np.zeros(sr), sr)

transcript = "I am going to kill and beat you to death."

print("Loading MFusTSVD-TA model...")
model = MFusTSVD_TA()
model.eval()

# -------------------------------------------------
# 4. Forward pass
# -------------------------------------------------
print("Running forward pass...")
with torch.no_grad():
    output = model(transcript, audio_path)

# -------------------------------------------------
# 5. Print results
# -------------------------------------------------
print("Model output:", output)
print("Output shape:", output.shape)

# -------------------------------------------------
# 6. Interpret prediction
# -------------------------------------------------
label_map = {
    0: "Safe",
    1: "Violence",
    2: "Sexual"
}

predicted_class = output.argmax(dim=1).item()
print("Predicted class:", label_map[predicted_class])

print("TEST COMPLETED SUCCESSFULLY ✅")
