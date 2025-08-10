# Whisper 日本語 音声認識デモ（Streamlit）

日本語の音声を可視化・文字起こしし、形態素解析まで行うデモアプリです。

## 機能
- 音声の波形可視化（Plotly）
- Whisper による日本語文字起こし（モデル選択: tiny/base/small/medium）
- モデル設定情報の表示
- 日本語形態素解析（spaCy ja_core_news_sm）
- デモ用日本語音声を自動生成（gTTS）

## セットアップ（uv）
```bash
cd /home/mutomasa/research_engineering/mutomasa/whisper_app
uv sync
```
（不足がある場合の追加）
```bash
uv add streamlit openai-whisper librosa soundfile plotly numpy pandas spacy gTTS \
  && uv add https://github.com/explosion/spacy-models/releases/download/ja_core_news_sm-3.8.0/ja_core_news_sm-3.8.0-py3-none-any.whl
```

## 起動
```bash
uv run streamlit run app.py --server.port 8503
```
ブラウザが自動で開かない場合は `http://localhost:8503` へアクセス。

## 使い方
1. サイドバーで以下を設定
   - 「デモ用の日本語音声を使う」をON（gTTSで自動生成）または音声ファイルをアップロード（.wav/.mp3）
   - Whisperモデル（tiny/base/small/medium）
2. 「解析を実行」をクリック
3. 画面内に波形・モデル情報・文字起こし・形態素解析（トークン/品詞/係り受け）が表示されます

## ファイル構成
```
whisper_app/
├── app.py        # Streamlitアプリ本体
├── .venv/        # uv仮想環境（自動作成）
└── README.md
```

## ヒント・注意
- gTTSはネット接続が必要です。接続不可の場合は手元の音声をアップロードしてください。
- GPU利用時にCUDA関連の警告が出る場合がありますが、CPUでも動作します。
- spaCy日本語モデルが見つからない場合は、上記の追加コマンドで `ja_core_news_sm` を導入してください。

## ライセンス
MIT
