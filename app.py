import gradio as gr 
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pywt
import os
import json
import random
import tempfile
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

SAMPLE_RATE = 22050
MAX_DURATION = 5
TIME_STEPS = 20
USE_DENOISE = True

model = load_model("Huan_luyen_6_huhong.h5")

def load_scaler_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(data['mean_'])
    scaler.scale_ = np.array(data['scale_'])
    scaler.n_features_in_ = len(scaler.mean_)
    return scaler

scaler = load_scaler_from_json("scaler.json")

with open("label_map.json", "r") as f:
    label_map = json.load(f)
index_to_label = {v: k for k, v in label_map.items()}

def denoise_wavelet(signal, wavelet='db8', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_denoised = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs_denoised, wavelet)

def create_sequences(mfcc, time_steps=20):
    return np.array([mfcc[i:i+time_steps] for i in range(len(mfcc) - time_steps)])

def cat_2s_ngau_nhien(y, sr, duration=2):
    if len(y) < duration * sr:
        return y
    start = random.randint(0, len(y) - duration * sr)
    return y[start:start + duration * sr]

def tao_anh_mel(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    y = cat_2s_ngau_nhien(y, sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(6, 3))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    ax.set_title("Phổ tần Mel", fontsize=10)
    fig.colorbar(img, ax=ax)
    plt.tight_layout()
    path = os.path.join(tempfile.gettempdir(), "mel.png")
    fig.savefig(path, dpi=80)
    plt.close()
    return Image.open(path)

def tao_wavelet_transform(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    y = cat_2s_ngau_nhien(y, sr)
    coef, _ = pywt.cwt(y, scales=np.arange(1, 128), wavelet='morl', sampling_period=1/sr)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(np.abs(coef), extent=[0, len(y)/sr, 1, 128], cmap='plasma', aspect='auto', origin='lower')
    ax.set_title("Phổ sóng con (Wavelet)")
    ax.set_xlabel("Thời gian (s)")
    ax.set_ylabel("Tần số (scale)")
    plt.tight_layout()
    path = os.path.join(tempfile.gettempdir(), "wavelet.png")
    fig.savefig(path, dpi=80)
    plt.close()
    return Image.open(path)

def tao_waveform_image(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    y = cat_2s_ngau_nhien(y, sr)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='steelblue')
    ax.set_title("Biểu đồ Sóng Âm (Waveform)")
    ax.set_xlabel("Thời gian (s)")
    ax.set_ylabel("Biên độ")
    plt.tight_layout()
    path = os.path.join(tempfile.gettempdir(), "waveform.png")
    fig.savefig(path, dpi=80)
    plt.close()
    return Image.open(path)

def bao_san_sang(file_path):
    if not file_path:
        return ""
    return "<b style='color:green;'>✅ Âm thanh đã sẵn sàng. Nhấn kiểm tra ngay!</b>"

def sinh_anh(file_path):
    if not file_path:
        return None, None, None
    mel_img = tao_anh_mel(file_path)
    wavelet_img = tao_wavelet_transform(file_path)
    waveform_img = tao_waveform_image(file_path)
    return mel_img, wavelet_img, waveform_img

def du_doan(file_path):
    if not file_path:
        return "<b style='color:red;'>❌ Chưa có âm thanh.</b>"

    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    signal, _ = librosa.effects.trim(signal)
    signal = librosa.util.fix_length(signal, size=SAMPLE_RATE * MAX_DURATION)

    if USE_DENOISE:
        signal = denoise_wavelet(signal)

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).T
    mfcc = scaler.transform(mfcc)
    X_input = create_sequences(mfcc, time_steps=TIME_STEPS)

    if len(X_input) == 0:
        return "<b style='color:red;'>⚠️ Âm thanh quá ngắn để phân tích.</b>"

    y_preds = model.predict(X_input, verbose=0)
    avg_probs = np.mean(y_preds, axis=0)
    pred_index = np.argmax(avg_probs)
    confidence = avg_probs[pred_index] * 100
    pred_label = "HƯ HỎNG KHÁC" if confidence < 60 else index_to_label[pred_index]

    html = f"""<div style='background:#f0faff;color:#000;padding:10px;border-radius:10px'>
<b style='color:#000'>📋 Kết Quả:</b><br>
✅ <b style='color:#000'>Tình trạng:</b> <span style='color:#007acc;font-size:18px'>{pred_label.upper()}</span><br>
📊 <b style='color:#000'>Độ tin cậy:</b> <span style='color:#000'>{confidence:.2f}%</span><br>
<hr style='margin:6px 0'>
<b style='color:#000'>Xác suất từng lớp:</b><br>"""
    for i, prob in enumerate(avg_probs):
        html += f"<span style='color:#000'>- {index_to_label[i]}: {prob*100:.1f}%</span><br>"
    html += "</div>"
    return html

def reset_output():
    return "", None, None, None, ""

def chon_file(f1, f2):
    return f1 if f1 else f2

with gr.Blocks(css="""
#check-btn {
    background: #007acc;
    color: white;
    height: 48px;
    font-size: 16px;
    font-weight: bold;
    border-radius: 10px;
}
""") as demo:

    gr.HTML("""
    <div style="
        display: flex;
        align-items: center;
        background-image: url('https://cdn-uploads.huggingface.co/production/uploads/6881f05ad0fc87fca019ee65/t7NwSiUHpjoFXh1S10MT4.png');
        background-repeat: no-repeat;
        background-size: 100px 40px;
        background-position: 0px 0px;   
        padding-left: 60px;
        height: 50px;
        margin: 0;                     
    ">
    </div>
    """)

    gr.Markdown("""
    <div style='
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: -10px;
        margin-bottom: 10px;
        height: 40px;
    '>
        <h4 style='color:#007acc; font-size:20px; font-weight:bold; margin: 0;'>
                  CHẨN ĐOÁN HƯ HỎNG TỪ ÂM THANH ĐỘNG CƠ 
        </h4>
    </div>
    """)

    with gr.Row():
        audio_file = gr.Audio(type="filepath", label="📂 Tải File Âm Thanh", interactive=True)
        audio_mic = gr.Audio(type="filepath", label="🎤 Ghi Âm", sources=["microphone"], interactive=True)

    thong_bao_ready = gr.HTML()
    btn_check = gr.Button("🔍 KIỂM TRA NGAY", elem_id="check-btn")
    output_html = gr.HTML()

    with gr.Accordion("📊 Phân tích Âm Thanh", open=False):
        mel_output = gr.Image(label="")
        wavelet_output = gr.Image(label="")
        waveform_output = gr.Image(label="")

    def xu_ly_toan_bo(file_path):
        tb = bao_san_sang(file_path)
        mel, wavl, wave = sinh_anh(file_path)
        kq = du_doan(file_path)
        return tb, mel, wavl, wave, kq

    audio_file.change(
        fn=xu_ly_toan_bo,
        inputs=audio_file,
        outputs=[thong_bao_ready, mel_output, wavelet_output, waveform_output, output_html]
    )

    audio_mic.change(
        fn=xu_ly_toan_bo,
        inputs=audio_mic,
        outputs=[thong_bao_ready, mel_output, wavelet_output, waveform_output, output_html]
    )

    btn_check.click(
        fn=lambda f1, f2: du_doan(chon_file(f1, f2)),
        inputs=[audio_file, audio_mic],
        outputs=output_html
    )

    audio_file.clear(fn=reset_output, outputs=[
        thong_bao_ready,
        mel_output,
        wavelet_output,
        waveform_output,
        output_html
    ])
    audio_mic.clear(fn=reset_output, outputs=[
        thong_bao_ready,
        mel_output,
        wavelet_output,
        waveform_output,
        output_html
    ])

demo.launch()
