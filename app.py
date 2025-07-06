"""
Unlock Your Full Potential – MVP  (pełna wersja / Krok 1-6)
============================================================
• Ręczny wpis snu  ➜ historia + wykresy
• Upload zdjęcia   ➜ analiza postawy (MediaPipe) + metryki
• AI-roadmapa 4-tyg. (GPT-4o)
• Future Body (obraz 1024×1024 z DALL·E 3 – szybka wersja, bez fine-tune)
------------------------------------------------------------
Wymagane pakiety w aktywnym .venv:
pip install streamlit pandas matplotlib pillow numpy \
            opencv-python-headless mediapipe openai python-dotenv pytest
.env obok app.py z kluczem:
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""
from datetime import date
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import openai

from src.data_loader import build_sleep_df
from src.pose_analysis import detect_keypoints, compute_metrics
from src.llm_coach   import build_roadmap   # GPT-4o helper

# ─── OpenAI init ────────────────────────────────────────────────────────────
load_dotenv()
client = openai.OpenAI()  # działa, jeśli OPENAI_API_KEY w .env

def generate_future_body(goal: str = "athletic, defined shoulders") -> str:
    """
    Ekspresowa wersja – tylko prompt → DALL·E 3 (bez in-paintingu).
    Zwraca URL obrazu 1024×1024.
    """
    prompt = (
        "Full-body studio photo of **the same person** but "
        f"{goal}. Natural lighting, 4K sharpness, no text."
    )
    resp = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    return resp.data[0].url

# ─── Streamlit UI ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Unlock Your Full Potential", page_icon="🏋️‍♂️")
st.title("🏋️‍♂️ Unlock Your Full Potential – MVP")

# 1️⃣ Formularz snu ----------------------------------------------------------
with st.expander("1️⃣ Wpisz dane snu"):
    c1, c2 = st.columns(2, gap="large")
    with c1:
        night    = st.date_input("Data snu", value=date.today())
        total_h  = st.number_input("Całkowity sen (h)", 0.0, 12.0, 7.0, 0.25)
        hrv      = st.number_input("HRV (ms)", 0, 300, 75)
    with c2:
        rem   = st.number_input("REM (min)",   0, 240, 90)
        deep  = st.number_input("Głęboki (min)", 0, 240, 60)
        light = st.number_input("Lekki (min)",  0, 420, 250)

    if st.button("Zapisz noc"):
        row = build_sleep_df(night, total_h, rem, deep, light, hrv)
        st.success("✅ Dodano wpis snu!")
        st.dataframe(row)

        hist = st.session_state.setdefault("sleep_history", pd.DataFrame())
        st.session_state["sleep_history"] = pd.concat([hist, row], ignore_index=True)

# 2️⃣ Upload zdjęcia + analiza postawy ---------------------------------------
img_file = st.file_uploader("2️⃣ Wgraj zdjęcie sylwetki (front)", ["jpg", "jpeg", "png"])
issues_for_llm: list[str] = []

if img_file:
    pil = Image.open(img_file).convert("RGB")
    cv  = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    try:
        annotated, pts = detect_keypoints(cv)
        m = compute_metrics(pts)

        st.subheader("🔍 Analiza postawy")
        st.image(annotated, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Asymetria barków", f"{m['shoulder_asym_px']:.0f} px")
        col2.metric("WHR (talia/biodra)", f"{m['waist_hip_ratio']:.2f}")

        if m["shoulder_asym_px"] > 20:
            st.warning("👉 Barki na różnych wysokościach")
            issues_for_llm.append("shoulder_asymmetry")
        if m["waist_hip_ratio"] > 1.1:
            st.info("💡 WHR wysoki – popraw proporcje")
            issues_for_llm.append("high_whr")
    except ValueError as e:
        st.error(str(e))

# 3️⃣ Wykresy snu ------------------------------------------------------------
avg_sleep = None
if "sleep_history" in st.session_state and not st.session_state["sleep_history"].empty:
    st.subheader("📊 Analiza snu")
    hist = st.session_state["sleep_history"]

    # HRV line
    fig1, ax1 = plt.subplots()
    ax1.plot(hist["date"], hist["hrv"], marker="o")
    ax1.set_ylabel("HRV [ms]")
    ax1.set_title("HRV w czasie")
    st.pyplot(fig1)

    # Sleep phases bar
    fig2, ax2 = plt.subplots()
    hist[["rem_min", "deep_min", "light_min"]].set_index(hist["date"]).plot(
        kind="bar", stacked=True, ax=ax2
    )
    ax2.set_ylabel("Minuty")
    ax2.set_title("Fazy snu")
    st.pyplot(fig2)

    avg_sleep = hist["total_sleep_h"].mean()
    st.markdown(f"**Średnia długość snu:** {avg_sleep:.1f} h")

# 4️⃣ Roadmapa GPT-4o --------------------------------------------------------
st.subheader("🗺️ Roadmapa (4 tyg.)")

if st.button("🧠 Generuj roadmapę"):
    if avg_sleep is None:
        st.error("Dodaj przynajmniej jeden wpis snu.")
    else:
        with st.spinner("GPT-4o tworzy plan…"):
            roadmap = build_roadmap(avg_sleep, issues_for_llm)
        st.markdown(roadmap, unsafe_allow_html=True)

# 5️⃣ Future Body ------------------------------------------------------------
st.subheader("🖼️ Future You")

if img_file and st.button("🔮 Pokaż przyszłą sylwetkę"):
    with st.spinner("Generuję obraz…"):
        url = generate_future_body()
    st.image(url, caption="Ty za ~6 mies. (wizja AI)", use_container_width=True)
