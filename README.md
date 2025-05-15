# Privacy-Preserving YOLOv5 Detection

This repository provides a complete pipeline for privacy-preserving rice disease detection using YOLOv5. It incorporates **differential privacy**, **model pruning**, and **retrieval-augmented generation (RAG)** to ensure secure and efficient disease identification and alerting.

---

## 🔍 Overview

The goal of this project is to detect common rice plant diseases using computer vision techniques while ensuring data privacy during training. The model is optimized for deployment and integrates a lightweight retrieval-based large language model to generate alerts based on detection results.

---

## 🚀 Features

- ⚡ YOLOv5-based object detection (detects: `rice_blast`, `brown_spot`, `bacterial_leaf_blight`)
- 🔐 Differential Privacy (DP-SGD) training using **Opacus**
- 🧠 Model pruning for compression and faster inference
- 🤖 Retrieval-Augmented Generation (RAG) using fine-tuned LLaMA-2
- 📦 Organized and modular code structure
- 📊 Performance evaluated using mAP and F1-score

---

## 📁 Project Structure

