
#  Multimodal Attention System for Video Question Answering  
*A deep learning model integrating visual & textual modalities using cross-modal attention.*

## Overview  
This project presents an **end-to-end Multimodal Video Question Answering (Video-QA) system** that jointly processes **video frames, subtitles, and natural-language questions**.  
The model integrates **ResNet50**, **BERT**, **BiLSTM**, and a **Cross-Modal Attention Fusion Module** to reason over temporal and semantic relations across modalities.

The system is trained and evaluated on the **TVQA dataset**, achieving a **test accuracy of 72.68%**, outperforming common multimodal baselines.

## Key Features  
- **Visual Encoder:**  
  - ResNet50 + temporal multi-head attention  
  - Optical flow–based motion modeling  
- **Textual Encoder:**  
  - BERT for questions, answers, and subtitles  
  - BiLSTM for subtitle temporal context  
- **Cross-Modal Fusion:**  
  - Video ↔ Subtitle  
  - Video ↔ Question  
  - Subtitle ↔ Question  
  - Combined through a 3-way cross-attention module  
- **Answer Prediction:**  
  - Similarity scoring + classification head  
  - Focal Loss + Label Smoothing  
- **Efficient Training:**  
  - AdamW optimizer  
  - Cosine learning rate schedule  
  - Mixed-precision training

## Architecture  
```
Video Frames ──► ResNet50 ──► Temporal Attention ──┐
                                                   │
Subtitles ─► BERT ─► BiLSTM ───────────────────────┼─► Cross-Modal Attention ──► Fusion MLP ─► Answer Head
                                                   │
Question + Options ──► BERT ───────────────────────┘
```

## Project Structure  
```
├── data/                     # TVQA dataset (frames, subtitles, QA pairs)
├── models/                   # Encoders, cross-modal attention, classifier
├── utils/                    # Preprocessing, metrics, loaders
├── experiments/              # Training logs, configs, checkpoints
├── main.py                   # Training/evaluation entry point
└── README.md                 # Documentation
```

## Results  
| Metric | Value |
|--------|--------|
| **Test Accuracy** | **72.68%** |
| **Best Validation Accuracy** | 71.00% |
| **Epochs** | 30 |
| **Validation Std. Dev (last 5 epochs)** | 0.0249 |
| **Parameters** | ~241M |

## Dataset (TVQA)  
- 4,706 video clips  
- 4,706 subtitle files  
- 22,395 QA pairs  
- 5-option multiple choice  
- Frames sampled at 1 fps (~290 frames per clip)

Preprocessing includes:  
- Frame resizing (224×224)  
- Subtitle–frame alignment  
- BERT tokenization (max 256 tokens)

## Training Configuration  
- **Batch Size:** 4 / 16  
- **Optimizer:** AdamW  
- **LR:** 3×10⁻⁵ (Cosine Annealing)  
- **Loss:** Focal Loss + Label Smoothing  
- **Hardware:** RTX 4050 (6 GB), Intel i5, 8 GB shared RAM  
- **Total Training Time:** ~8 hours (30 epochs)

## Tech Stack  
- Python 3.10  
- PyTorch 2.2  
- Transformers (HuggingFace)  
- OpenCV  
- NumPy / Pandas  
- Matplotlib  

## Future Improvements  
- Vision-Language Transformers for deeper fusion  
- 3D CNN / Video-Transformer temporal modeling  
- Multimodal data augmentation  
- Model compression for edge deployment  
- Attention map visualization for explainability  

## Authors  
**Batch – B | Team – 7**  
- Jainithissh S  
- Nitheshkummar C  
- Akhilesh Kumar S  
- Krishna Prakash S  

## Citation  
```
@article{MultimodalVQA2025,
  title={Multimodal Attention System for Video Question Answering},
  author={Jainithissh S and Team B-7},
  year={2025}
}
```
