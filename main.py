from faster_whisper import WhisperModel
from datetime import datetime

def ASR(devices:str, audio_path:str):
    model_size = "large-v3"
    print(f"Model : {model_size}")
    model = WhisperModel(
        model_size
        , device=devices
        , compute_type="float16" if devices == "cuda" else "int8")
    print(f"Whisper : {model}")
    start = datetime.now().timestamp()
    print(f"start : {start}")
    segments, info = model.transcribe(
        audio_path,
        language="th",
        beam_size=10,
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
        # initial_prompt="",
        word_timestamps=True,
        temperature=0.1,
    )

    print(
        f"Detected language: '{info.language}' with probability {info.language_probability:.4f}"
    )
    print("-" * 60)

    for segment in segments:
        lang = getattr(segment, "language", info.language) or info.language
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] ({lang})")
        print(f"  {segment.text.strip()}")
        print()

    end = datetime.now().timestamp()

    print("-" * 60)
    print(f"Total time: {end-start:.2f} seconds")


ASR('cpu', "/Users/dechthanaarunchaiya/Desktop/ForGLS/ForSTT/Test_STT/Soi Seri Thai 1-2.mp3")