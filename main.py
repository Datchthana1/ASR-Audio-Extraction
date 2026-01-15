from faster_whisper import WhisperModel
from datetime import datetime

# ใช้ large-v3 สำหรับ multilingual ที่ดีที่สุด
model_size = "large-v3"
print(f"Model : {model_size}")

model = WhisperModel(
    model_size,
    device="cpu",
    compute_type="int8"
)

print(f"Whisper : {model}")

start = datetime.now().timestamp()
print(f"start : {start}")

# Multilingual transcription settings
# หมอ: ภาษาไทย, ผู้ป่วย: หลายภาษา
segments, info = model.transcribe(
    "/Users/dechthanaarunchaiya/Desktop/ForGLS/ForSTT/Soi Lat Phrao 111 4.mp3",
    language="th",
    beam_size=10,
    best_of=5,
    vad_filter=True,
    vad_parameters={
        "threshold": 0.2,              # ลดลงเพื่อจับเสียงเบา
        "min_speech_duration_ms": 100, # จับ speech สั้นๆ ได้
        "min_silence_duration_ms": 200,
        "speech_pad_ms": 500,          # เพิ่ม padding
    },

    # Quality settings - ผ่อนคลายลงเพื่อไม่ให้ตัดทิ้ง
    log_prob_threshold=-1.0,
    no_speech_threshold=0.4,
    compression_ratio_threshold=2.8,

    # ปิด repetition penalty เพราะอาจทำให้คำซ้ำหาย
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,

    # Context
    condition_on_previous_text=False,  # ปิดเพื่อไม่ให้ error สะสม

    # Prompt ภาษาไทยล้วน - ใส่คำที่มักถอดผิด
    initial_prompt="บทสนทนาในคลินิก หมอถามผู้ป่วย ครับ ค่ะ อาการ ไข้ เวียนหัว ปวดหัว คลื่นไส้ ตรวจสุขภาพ อุณหภูมิ วัดไข้ ประวัติสุขภาพ",

    # Word-level timestamps
    word_timestamps=True,

    # Temperature - ลดความ creative
    temperature=0.0,
)

print(f"Detected language: '{info.language}' with probability {info.language_probability:.4f}")
print("-" * 60)

for segment in segments:
    # แสดงภาษาของแต่ละ segment (ถ้ามี)
    lang = getattr(segment, 'language', info.language) or info.language
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] ({lang})")
    print(f"  {segment.text.strip()}")
    print()

end = datetime.now().timestamp()

print("-" * 60)
print(f"Total time: {end-start:.2f} seconds")

# import torch
# from transformers import pipeline

# MODEL_NAME = "biodatlab/whisper-th-medium-combined"  # see alternative model names below
# lang = "th"
# device = 0 if torch.cuda.is_available() else "cpu"
# pipe = pipeline(
#     task="automatic-speech-recognition",
#     model=MODEL_NAME,
#     chunk_length_s=30,
#     device=device,
# )

# # Perform ASR with the created pipe.
# pipe("/Users/dechthanaarunchaiya/Desktop/ForGLS/ForSTT/Thanya Park.mp3", generate_kwargs={"language":"<|th|>", "task":"transcribe"}, batch_size=16)["text"]