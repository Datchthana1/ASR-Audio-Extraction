from faster_whisper import WhisperModel
from datetime import datetime
import pyaudio as pa
import numpy as np

p = pa.PyAudio()

channels = 1
rate = 16000
input = True
frames_per_buffer = 1024

def model(model_size, devices):
    model = WhisperModel(
        model_size
        , device=devices
        , compute_type="float16" if devices == "cuda" else "int8")
    print(f"Model : {model_size}")
    return model

def sound_devices(rate, frames_per_buffer, model):
    stream = p.open(
        format=pa.paInt16,
        channels=1,
        rate=rate,
        input=True,
        frames_per_buffer=frames_per_buffer
    )

    buffer = []
    samples_collection = 0
    print(f"INFO : Start Recording")
    while True:
        data = stream.read(frames_per_buffer, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        audio /= 32768.0
        buffer.append(audio)
        samples_collection += len(audio)
        if samples_collection >= rate:
            chunk_audio = np.concatenate(buffer)
            segments, _ = model.transcribe(
                    chunk_audio,
                    language="th",
                    beam_size=1,
                    best_of=5,
                    vad_filter=True,
                    vad_parameters={
                        "threshold": 0.2,
                        "min_speech_duration_ms": 100,
                        "min_silence_duration_ms": 200,
                        "speech_pad_ms": 500,
                    },
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.4,
                    compression_ratio_threshold=2.8,
                    repetition_penalty=1.0,
                    no_repeat_ngram_size=0,
                    condition_on_previous_text=False,
            #         initial_prompt="""
            #             บทสนทนาเกี่ยวกับการสัมพาณษ์ผู้ป่ยเกี่ยวกับโรคที่ผู้ป่วยเข้ารับการบริการไม่ว่าจะเป็นการเข้ามาบริการต่างๆ
            #             ซึ่งจะเกี่ยวข้องกับภาษาทางการแพทย์ต่างๆ ไม่ว่าจะเป็นการสอบถามอากจารย์ที่จำเพาะเจาะจง หรือรวมถึงประวัติผู้ป่วยที่เข้ารับการรักษาด้วยอาการต่างๆ
            # """,
                    word_timestamps=True,
                    temperature=0.0,
            )
            for seg in segments:
                print(seg.text, end=' ', flush=True)
            
            buffer = []
            samples_collection = 0


Wmodel = model('large-v3', 'cpu')
sound_devices(rate=rate, frames_per_buffer=frames_per_buffer, model=Wmodel)