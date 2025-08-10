import io
import os
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import spacy
import streamlit as st
import whisper
from gtts import gTTS


@st.cache_resource
def load_whisper(model_name: str = "small"):
    return whisper.load_model(model_name)


@st.cache_resource
def load_nlp():
    try:
        return spacy.load("ja_core_news_sm")
    except Exception:
        return None


def ensure_demo_audio() -> str:
    demo_dir = Path("demo_audio"); demo_dir.mkdir(exist_ok=True)
    target = demo_dir / "demo_ja_gtts.wav"
    if not target.exists():
        # gTTSで日本語音声を合成
        text = "こんにちは。これは日本語の音声認識デモです。ランダムフォーレストのアニメーションも見てください。"
        tts = gTTS(text=text, lang="ja")
        mp3_path = demo_dir / "demo_ja_gtts.mp3"
        tts.save(str(mp3_path))
        # MP3 → WAV 変換
        y, sr = librosa.load(str(mp3_path), sr=16000)
        sf.write(str(target), y, sr)
    return str(target)


def plot_waveform(y: np.ndarray, sr: int):
    t = np.arange(len(y)) / sr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name="waveform"))
    fig.update_layout(xaxis_title="Time [s]", yaxis_title="Amplitude", margin=dict(l=0, r=0, t=10, b=0))
    return fig


def main():
    st.set_page_config(page_title="Whisper 日本語 音声認識デモ", layout="wide")
    st.title("🗣️ Whisper: 日本語 音声認識 + 可視化 + 形態素解析")

    with st.sidebar:
        st.subheader("入力")
        use_demo = st.checkbox("デモ用の日本語音声を使う", value=True)
        uploaded = st.file_uploader("音声ファイルをアップロード (.wav/.mp3)", type=["wav","mp3"])
        model_size = st.selectbox("Whisperモデル", ["tiny","base","small","medium"], index=2)
        run_btn = st.button("🔍 解析を実行")

    audio_path = None
    if run_btn:
        if use_demo:
            audio_path = ensure_demo_audio()
        elif uploaded is not None:
            tmp = Path("tmp"); tmp.mkdir(exist_ok=True)
            audio_path = str(tmp / uploaded.name)
            with open(audio_path, "wb") as f:
                f.write(uploaded.read())

        if audio_path is None:
            st.warning("音声が選択されていません")
            return

        y, sr = librosa.load(audio_path, sr=16000)
        st.subheader("波形")
        st.plotly_chart(plot_waveform(y, sr), use_container_width=True)

        st.subheader("Whisper モデル情報")
        st.write({"selected_model": model_size})

        st.subheader("認識結果（日本語）")
        asr = load_whisper(model_size)
        res = asr.transcribe(audio_path, language="ja")
        text = res.get("text", "")
        st.text_area("transcript", value=text, height=120)

        st.subheader("形態素解析（spaCy日本語）")
        nlp = load_nlp()
        if nlp is None:
            st.info("spaCy日本語モデルが見つかりませんでした。READMEの手順で ja_core_news_sm を導入してください。")
        else:
            doc = nlp(text)
            df_rows = [{"text": t.text, "pos": t.pos_, "dep": t.dep_} for t in doc]
            st.dataframe(df_rows, use_container_width=True)


if __name__ == "__main__":
    main()


