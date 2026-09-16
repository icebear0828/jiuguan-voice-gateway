import os
import sys
import time
import numpy as np
import soundfile as sf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", PROJECT_ROOT)
SOVITS_ROOT = os.environ.get("SOVITS_ROOT", os.path.abspath(os.path.join(PROJECT_ROOT, "..", "GPT-SoVITS")))
SOVITS_CORE = os.path.join(SOVITS_ROOT, "GPT_SoVITS")
sys.path.insert(0, SOVITS_ROOT)
sys.path.insert(0, SOVITS_CORE)
sys.path.insert(0, os.path.join(SOVITS_CORE, "eres2net"))

if os.path.exists(SOVITS_ROOT):
    os.chdir(SOVITS_ROOT)

from TTS_infer_pack.TTS import TTS, TTS_Config

MODEL_DIR = os.path.join(WORKSPACE_DIR, "assets/models/rimuru/v2pro_master")
REF_AUDIO = os.path.join(MODEL_DIR, "reference_audio/01_Master_Daily.wav")
REF_TEXT = "自己紹介がまだだったな 俺はリムル＝テンペスト"

cfg = TTS_Config()
cfg.bert_base_path = os.path.join(SOVITS_CORE, "pretrained_models/chinese-roberta-wwm-ext-large")
cfg.cnhuhbert_base_path = os.path.join(SOVITS_CORE, "pretrained_models/chinese-hubert-base")
cfg.t2s_weights_path = os.path.join(MODEL_DIR, "Rimuru_v2Pro_Native-e10.ckpt")
cfg.vits_weights_path = os.path.join(MODEL_DIR, "Rimuru_v2Pro_e25_s6825.pth")
cfg.device = "cpu"
cfg.is_half = False

print("Loading TTS...")
tts = TTS(cfg)
print("Loaded. Running inference...")

req = {
    "text": "俺はスライムのリムル！悪いスライムじゃないよ！",
    "text_lang": "ja",
    "ref_audio_path": REF_AUDIO,
    "prompt_text": REF_TEXT,
    "prompt_lang": "ja",
    "top_k": 5,
    "top_p": 0.95,
    "temperature": 0.70,
    "text_split_method": "cut0",
    "sample_steps": 32,
    "speed_factor": 1.0
}

t0 = time.time()
gen = tts.run(req)
sr, chunk = next(gen)
t1 = time.time()

if chunk.dtype == np.int16:
    chunk_f = chunk.astype(np.float32) / 32768.0
else:
    chunk_f = chunk.astype(np.float32)

peak = np.max(np.abs(chunk_f))
if peak > 0:
    chunk_f = (chunk_f / peak) * 0.8410 # -1.5 dBFS

out_path = os.path.join(WORKSPACE_DIR, "scratch/rimuru_test_quick.wav")
sf.write(out_path, chunk_f, sr, subtype="PCM_16")
print(f"Generated test audio in {t1 - t0:.2f}s: {out_path} (SR: {sr})")
