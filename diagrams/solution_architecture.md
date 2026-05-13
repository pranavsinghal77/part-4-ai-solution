# Solution Architecture — Manufacturing Visual Defect Inspection

## System Flow

```
┌──────────────────────────────────────────────────────────────┐
│                   PRODUCTION LINE                            │
│                                                              │
│  [Conveyor Belt] ──→ [Industrial Camera (RGB, 20fps)]        │
└─────────────────────────────┬────────────────────────────────┘
                              │ Raw image frame
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREPROCESSING MODULE                      │
│  • Resize to 224×224                                        │
│  • Normalize pixel values [0, 1]                            │
│  • Optional: brightness/contrast correction                 │
└─────────────────────────────┬───────────────────────────────┘
                              │ Preprocessed tensor
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              AI INFERENCE ENGINE                            │
│  Model: EfficientNet-B0 (TF SavedModel / ONNX)             │
│  Hardware: NVIDIA Jetson edge device or cloud endpoint      │
│  Latency: < 50ms per image                                  │
│                                                             │
│  Output: {normal: 0.02, scratch: 0.91, dent: 0.04, stain: 0.03} │
└──────┬────────────────────────────┬────────────────────────-┘
       │ Confidence ≥ 0.85          │ Confidence < 0.85
       ▼                            ▼
┌─────────────┐            ┌──────────────────────┐
│ AUTOMATED   │            │  HUMAN REVIEW QUEUE  │
│ PLC ACTION  │            │  (QC Supervisor App) │
│             │            │  Inspector confirms  │
│ NORMAL → ✓  │            │  or overrides        │
│ DEFECT → ✗  │            └──────────────────────┘
└──────┬──────┘
       │ All decisions (AI + human)
       ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATA & MONITORING LAYER                    │
│                                                             │
│  • Real-time dashboard (defect rate, throughput, alerts)   │
│  • Data warehouse (all predictions + labels stored)        │
│  • Weekly calibration report (AI vs. human audit)          │
│  • Drift detector (confidence distribution monitor)        │
│  • Retraining trigger (if accuracy drops > 3%)             │
└─────────────────────────────────────────────────────────────┘
```

## Component Summary

| Component | Technology | Purpose |
|---|---|---|
| Camera | Industrial RGB camera | Capture product surface at line speed |
| Preprocessing | OpenCV / TF.image | Resize, normalize, augment |
| Inference Engine | EfficientNet-B0 | Classify defect type |
| Edge Device | NVIDIA Jetson AGX | Low-latency on-premise inference |
| PLC Integration | Siemens S7 / OPC-UA | Trigger physical reject mechanism |
| Review App | React + FastAPI | Human-in-the-loop exception handling |
| Dashboard | Grafana / PowerBI | Real-time KPI monitoring |
| Data Warehouse | PostgreSQL / Snowflake | Store predictions + labels for retraining |
| MLOps | MLflow + Airflow | Model versioning, scheduled retraining |
