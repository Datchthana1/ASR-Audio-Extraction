from faster_whisper import WhisperModel
import os

# เช็คไฟล์
if not os.path.exists("audio.mp3"):
    print("❌ ไม่พบไฟล์ audio.mp3")
    exit()

print("⏳ กำลังโหลด model...")
model = WhisperModel("base", device="cpu", compute_type="int8")

print("⏳ กำลัง transcribe...")
segments, info = model.transcribe("audio.mp3", beam_size=5)

print(f"✅ ตรวจจับภาษา: {info.language} ({info.language_probability:.2f})")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")