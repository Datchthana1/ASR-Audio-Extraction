import jiwer
from pythainlp import word_tokenize
import re

def normalize_thai(text: str) -> str:
    """ทำให้ข้อความเป็นมาตรฐาน"""
    # ลบ whitespace ซ้ำ
    text = re.sub(r'\s+', ' ', text)
    # ลบเครื่องหมายวรรคตอน (ถ้าไม่สำคัญ)
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', '', text)
    # lowercase
    text = text.lower().strip()
    return text

def thai_word_tokenize(text: str) -> str:
    """
    ตัดคำภาษาไทยแล้วเว้นช่องว่าง
    นี่คือขั้นตอนสำคัญ!
    """
    words = word_tokenize(text, engine='newmm')
    return ' '.join(words)

def calculate_wer(reference: str, hypothesis: str):
    """
    WER สำหรับภาษาไทย
    """
    # **ต้องตัดคำก่อน**
    ref_tokenized = thai_word_tokenize(normalize_thai(reference))
    hyp_tokenized = thai_word_tokenize(normalize_thai(hypothesis))
    
    print(f"Reference (tokenized):  {ref_tokenized}")
    print(f"Hypothesis (tokenized): {hyp_tokenized}")
    print("-" * 60)
    
    # คำนวณ WER
    wer = jiwer.wer(ref_tokenized, hyp_tokenized)
    measures = jiwer.process_words(ref_tokenized, hyp_tokenized)
    
    print(f"Measures: {measures}")
    print("-" * 60)
    
    return {
        'WER': wer * 100,
        'MER': measures.mer * 100,
        'WIL': measures.wil * 100,
        'Hits': measures.hits,
        'Substitutions': measures.substitutions,
        'Deletions': measures.deletions,
        'Insertions': measures.insertions,
        'Total_words': len(ref_tokenized.split())
    }

# ตัวอย่าง
reference = "คูณมีอ้าการณขวดหัวมานานแค่ไหนหรือคับ"
hypothesis = "คุณมีอาการปวดหัวมานานแค่ไหนหรือครับคุณหญิง"

result = calculate_wer(reference, hypothesis)

print(f"WER: {result['WER']:.2f}%")
print(f"Total words: {result['Total_words']}")
print(f"Hits: {result['Hits']}")
print(f"Substitutions: {result['Substitutions']}")
print(f"Deletions: {result['Deletions']}")
print(f"Insertions: {result['Insertions']}")
# ```

# ---

# ## Output ที่ถูกต้อง
# ```
# Reference (tokenized):  คูณ มี อ้าการณ ขวด หัว มา นาน แค่ไหน หรือ คับ
# Hypothesis (tokenized): คุณ มี อาการ ปวด หัว มา นาน แค่ไหน หรือ ครับ คุณหญิง
# ------------------------------------------------------------
# Measures: WordOutput(
#     hits=6,                    # มา, หัว, นาน, แค่ไหน, หรือ, มี
#     substitutions=4,           # คูณ→คุณ, อ้าการณ→อาการ, ขวด→ปวด, คับ→ครับ
#     insertions=1,              # +คุณหญิง
#     deletions=0
# )
# ------------------------------------------------------------
# WER: 45.45%
# Total words: 11
# Hits: 6
# Substitutions: 4
# Deletions: 0
# Insertions: 1