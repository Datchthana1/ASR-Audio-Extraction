import librosa
import torch
from transformers import pipeline
from datetime import datetime

# ===== ตรวจสอบ CUDA =====
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
print("-" * 60)

# ===== Load Audio =====
print("Loading audio...")
audio_path = r"D:\OneFile\WorkOnly\AllCode\GLSWork\ForSTT\Soi Lat Phrao 111 4.mp3"
audio, sr = librosa.load(audio_path, sr=16000)
print(f"Audio loaded: {len(audio)/sr:.2f} seconds")
print("-" * 60)

# ===== Load Model =====
MODEL_NAME = "biodatlab/whisper-th-medium-combined"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model on {device}...")
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

print(f"Model device: {pipe.model.device}")
print("-" * 60)

# ===== Transcribe =====
print("Transcribing...")
start = datetime.now()

result = pipe(
    {"array": audio, "sampling_rate": sr},
    generate_kwargs={"language": "th", "task": "transcribe"},
    batch_size=16,
)

end = datetime.now()

# ===== Output =====
print("\n" + "=" * 60)
print("TRANSCRIPT:")
print("=" * 60)
print(result["text"])
print("=" * 60)
print(f"Processing time: {(end - start).total_seconds():.2f} seconds")

# ส่ง numpy array เข้า pipeline
result = pipe(
    {"array": audio, "sampling_rate": sr},
    generate_kwargs={"language": "th", "task": "transcribe"},
    batch_size=16,
)

print(result["text"])
