# 🎙️ Speech-to-Text Model with AI Summary and Denoising

An end-to-end speech processing application designed for **Indian-accented and Hindi-English code-mixed speech**. The system combines automatic speech recognition, audio denoising, sentiment analysis, text summarization, and subtitle generation into a single browser-based application.

---

## 📌 Project Overview

This project develops a practical **Automatic Speech Recognition (ASR)** pipeline using a pretrained **Whisper-small** model.

The system is designed to handle speech containing Indian accents and Hindi-English code switching. In addition to transcription, the application provides:

- 🎙️ Speech-to-text transcription
- 🔇 Optional audio denoising
- 😊 Sentiment analysis
- 📝 AI-based text summarization
- 🎬 SRT subtitle generation
- 🌐 Browser-based interactive interface
- ⚡ Parameter-efficient fine-tuning experiments using PEFT/LoRA

The application is implemented using **Python, Hugging Face Transformers, PEFT, Librosa, Noisereduce, VADER, BART, and Gradio**, and can be deployed using Hugging Face Spaces.

---

## 🎯 Objectives

- Develop an end-to-end speech-to-text system optimized for **Indian-accented and Hindi-English code-mixed speech**.
- Adapt a pretrained Whisper-small ASR model for speech recognition.
- Investigate **parameter-efficient model adaptation** using PEFT and LoRA.
- Reduce the number of trainable parameters through layer freezing and LoRA.
- Provide additional NLP capabilities such as sentiment analysis and summarization.
- Build a browser-accessible ML application for practical ASR deployment.

---

## 🏗️ System Architecture

```text
                    ┌──────────────────┐
                    │   Audio Input    │
                    │  WAV / MP3 etc.  │
                    └────────┬─────────┘
                             │
                             ▼
                 ┌──────────────────────┐
                 │  Audio Preprocessing │
                 │                      │
                 │ Librosa              │
                 │ Noisereduce (optional)│
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Whisper-small      │
                 │   ASR Model          │
                 │                      │
                 │ Fine-tuned / PEFT    │
                 │ LoRA Experiments     │
                 └──────────┬───────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ Transcription│
                     └──────┬───────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
        ┌─────────┐    ┌─────────┐    ┌──────────┐
        │  VADER  │    │  BART   │    │   SRT    │
        │Sentiment│    │Summary  │    │Subtitles │
        └────┬────┘    └────┬────┘    └─────┬────┘
             │              │               │
             └──────────────┼───────────────┘
                            ▼
                  ┌───────────────────┐
                  │    Gradio Web UI  │
                  └─────────┬─────────┘
                            │
                            ▼
                         User
