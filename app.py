import os
import time
import glob
import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image
from gtts import gTTS

# traducción estable (reemplaza a googletrans)
try:
    from deep_translator import GoogleTranslator
except ModuleNotFoundError:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deep-translator"])
    from deep_translator import GoogleTranslator

# ---------- utilidades ----------
def remove_files(days: int = 7):
    os.makedirs("temp", exist_ok=True)
    mp3_files = glob.glob("temp/*.mp3")
    if not mp3_files:
        return
    now = time.time()
    ttl = days * 86400
    for f in mp3_files:
        try:
            if os.stat(f).st_mtime < now - ttl:
                os.remove(f)
        except FileNotFoundError:
            pass

def safe_filename(s: str, maxlen: int = 20) -> str:
    base = (s or "audio").strip()[:maxlen]
    base = "".join(ch for ch in base if ch.isalnum() or ch in (" ", "-", "_")).strip()
    return base or "audio"

def translate_text(text: str, src: str, dest: str) -> str:
    if not text:
        return ""
    # deep-translator usa 'zh-CN'
    if dest.lower() == "zh-cn": dest = "zh-CN"
    if src.lower() == "zh-cn":  src  = "zh-CN"
    return GoogleTranslator(source=src, target=dest).translate(text)

def text_to_speech(input_language: str, output_language: str, text: str, tld: str):
    if not text:
        return None, "No se detectó texto para convertir."
    trans_text = translate_text(text, src=input_language, dest=output_language) or text
    tts = gTTS(trans_text, lang=output_language, tld=tld, slow=False)
    os.makedirs("temp", exist_ok=True)
    file_name = safe_filename(text)
    out_path = f"temp/{file_name}.mp3"
    tts.save(out_path)
    return file_name, trans_text

remove_files(7)

# ---------- UI ----------
st.title("Swiftie OCR — Caza de letras")
st.subheader("Elige la fuente de la imagen (cámara o archivo), detecta el texto y conviértelo en audio (Taylor’s Version)")

# Cámara / archivo
cam_ = st.checkbox("Usar cámara")
if cam_:
    img_file_buffer = st.camera_input("Toma una foto")
else:
    img_file_buffer = None

with st.sidebar:
    st.subheader("Procesamiento para cámara")
    filtro = st.radio("Filtro para imagen con cámara", ("Con filtro", "Sin filtro"))

bg_image = st.file_uploader("Cargar imagen:", type=["png", "jpg", "jpeg"])

# Texto extraído (variable compartida)
texto_detectado = ""

# ---- Si suben archivo
if bg_image is not None:
    st.image(bg_image, caption='Imagen cargada', use_container_width=True)
    # Leer bytes y decodificar con OpenCV sin escribir a disco
    bytes_data = bg_image.getvalue()
    cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    texto_detectado = pytesseract.image_to_string(img_rgb)
    st.write(texto_detectado)

# ---- Si usan cámara
if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    if filtro == "Con filtro":
        # inversión simple que a veces mejora contraste para OCR
        cv2_img = cv2.bitwise_not(cv2_img)

    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    texto_detectado = pytesseract.image_to_string(img_rgb)
    st.write(texto_detectado)

# ---------- Sidebar: traducción + TTS ----------
with st.sidebar:
    st.subheader("Parámetros de traducción y voz")

    lang_options = ["Inglés", "Español", "Bengali", "Coreano", "Mandarín", "Japonés"]
    lang_code = {
        "Inglés": "en",
        "Español": "es",
        "Bengali": "bn",
        "Coreano": "ko",
        "Mandarín": "zh-cn",
        "Japonés": "ja",
    }

    in_lang = st.selectbox("Lenguaje de entrada", lang_options, index=1)
    input_language = lang_code[in_lang]

    out_lang = st.selectbox("Lenguaje de salida", lang_options, index=0)
    output_language = lang_code[out_lang]

    english_accent = st.selectbox(
        "Acento (si eliges inglés como salida)",
        ("Default", "India", "United Kingdom", "United States", "Canada", "Australia", "Ireland", "South Africa"),
    )
    tld_map = {
        "Default": "com",
        "India": "co.in",
        "United Kingdom": "co.uk",
        "United States": "com",
        "Canada": "ca",
        "Australia": "com.au",
        "Ireland": "ie",
        "South Africa": "co.za",
    }
    tld = tld_map.get(english_accent, "com")

    display_output_text = st.checkbox("Mostrar texto traducido")

    if st.button("Convertir a audio"):
        file_id, output_text = text_to_speech(input_language, output_language, texto_detectado, tld)
        if file_id is None:
            st.warning(output_text)  # mensaje de error si no hay texto
        else:
            with open(f"temp/{file_id}.mp3", "rb") as audio_file:
                audio_bytes = audio_file.read()
            st.markdown("## Tu audio:")
            st.audio(audio_bytes, format="audio/mp3", start_time=0)

            if display_output_text:
                st.markdown("## Texto de salida:")
                st.write(output_text)
