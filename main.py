import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import os

# =====================================================
# 1. FASTAPI INIT
# =====================================================
app = FastAPI(title="Hate Speech API (Sensitive Version)")

class TextRequest(BaseModel):
    text: str


# =====================================================
# 2. LOAD MODEL (NO BASE MODEL, STANDALONE)
# =====================================================
MODEL_DIR = r"D:\DLClassifyRacist\model_indoberttweet"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🔄 Memuat model & tokenizer...")

if not os.path.exists(MODEL_DIR):
    raise RuntimeError(f"❌ Folder model tidak ditemukan: {MODEL_DIR}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    model.to(device)
    model.eval()

    # Optional: CPU optimization
    if device.type == "cpu":
        print("⚙️ Mengaktifkan dynamic quantization (CPU)")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    print(f"✅ Model siap di device: {device}")

except Exception as e:
    raise RuntimeError(f"❌ Gagal load model: {e}")


# =====================================================
# 3. PREDICTION FUNCTION
# =====================================================
def get_prediction(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    return probs[0]  # tensor [prob_non_hate, prob_hate]


# =====================================================
# 4. API ENDPOINT
# =====================================================
@app.post("/predict")
async def predict_text(request: TextRequest):
    start_time = time.time()

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Teks kosong")

    # ---- A. ORIGINAL PREDICTION ----
    probs_original = get_prediction(request.text)
    hate_score = probs_original[1].item()
    pred_idx = torch.argmax(probs_original).item()

    label_map = {
        0: "Bukan Ujaran Kebencian",
        1: "Ujaran Kebencian"
    }

    is_hate = pred_idx == 1
    trigger_details = []

    # ---- B. WORD IMPACT ANALYSIS ----
    if is_hate:
        words = request.text.split()
        words = words[:30]  # limit for performance

        impacts = []

        for i in range(len(words)):
            reduced_text = " ".join(words[:i] + words[i+1:])

            if not reduced_text.strip():
                continue

            probs_new = get_prediction(reduced_text)
            new_score = probs_new[1].item()

            drop = hate_score - new_score

            if drop > 0:
                impacts.append({
                    "word": words[i],
                    "confidence_without_word": round(new_score, 4),
                    "impact_drop": round(drop, 6)
                })

        impacts.sort(key=lambda x: x["impact_drop"], reverse=True)
        trigger_details = impacts[:5]

    process_time = time.time() - start_time

    return {
        "text": request.text,
        "prediction": label_map[pred_idx],
        "original_confidence": round(hate_score, 4),
        "is_hate_speech": is_hate,
        "trigger_analysis": trigger_details,
        "process_time_seconds": round(process_time, 3)
    }