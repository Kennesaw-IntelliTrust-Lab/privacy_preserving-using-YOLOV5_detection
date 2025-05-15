<div align="center">
  <a href="https://github.com/ultralytics/yolov5">
    <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLO banner">
  </a>
</div>

> ⚠️ **This project builds on [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)**  
> We extend it with **Differential Privacy (DP-SGD)**, **Model Pruning**, and a **Retrieval-Augmented Generation (RAG)** system using a fine-tuned **LLaMA-2 model** for agricultural disease alerting.

---

# 🛡️ Privacy-Preserving YOLOv5 Detection

This repository presents a YOLOv5-based system that detects **rice plant diseases** (Rice Blast, Brown Spot, Bacterial Leaf Blight) and generates privacy-preserving, real-time responses. It integrates:

- 🔐 Differential Privacy using Opacus
- ✂️ Model Compression via Structured Pruning
- 🤖 LLaMA-2-based RAG pipeline for natural language advice

---

## 📌 Highlights

- ⚙️ YOLOv5 fine-tuned on custom rice disease dataset
- 🔐 Differentially Private SGD (DP-SGD) with Opacus
- ✂️ Structured pruning to reduce model size by up to 30%
- 📘 RAG: LLaMA-2 + FAISS + curated `rag_docs.txt`
- 📊 Evaluation using mAP, F1-score, and ROUGE

---

## 📁 Project Structure

