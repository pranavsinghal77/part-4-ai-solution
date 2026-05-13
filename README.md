# Part 4: AI Solution Design for a Business Problem

| | |
|---|---|
| **Name** | Pranav Singhal |
| **Student ID** | BA2511208 |
| **Course** | Certification in Business Analytics with Gen & Agentic AI |
| **Part** | Part 4 of 4 — AI Solution Design |

---

## What This Project Does

In this part I worked as an AI Business Analyst rather than a model builder. The task was to pick a real-world business problem and design a complete AI solution — covering the problem definition, data requirements, model selection, evaluation strategy, and responsible AI risks. I chose the **manufacturing domain** because it connects naturally with the CNN work in Part 2.

---

## Reference Material Used

- **Source:** [Google Drive Dataset Folder](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)
- **Files referenced:** `ai_usecase_reference_catalog.csv`, `business_kpi_sample.csv`

---

## Repository Structure

```
part-4-ai-solution-design/
├── README.md
├── solution_report.md          ← Full 8-task design document
└── diagrams/
    └── solution_architecture.md
```

---

## Summary of My Design

**Domain:** Manufacturing  
**Problem:** Surface defect detection on production lines  
**AI Task:** Image Classification  
**Model:** CNN with transfer learning (EfficientNet-B0, ImageNet pre-trained)

### Why Manufacturing?
Manual visual inspection at the end of a conveyor belt is slow (2–4 items/sec max), fatigues inspectors, and misses 7–8% of defects. A camera-based CNN system can inspect 20+ items per second with consistent accuracy regardless of shift time or inspector experience.

### Solution at a Glance

```
Camera → Preprocessing → EfficientNet-B0 → Defect class + confidence
         ├── Confidence ≥ 0.85 → Automated PLC reject/pass signal
         └── Confidence < 0.85 → Human review queue
              ↓
         Dashboard + Data Warehouse (all predictions logged)
```

### Expected Business Impact

| KPI | Before AI | After AI (12 months) |
|---|---|---|
| Defect escape rate | ~7–8% | < 1% |
| Inspection throughput | 4 items/sec | 20+ items/sec |
| QC labour cost | Baseline | −40% |
| Customer defect complaints | Baseline | −60% |

### Responsible AI Highlights
- Low-confidence predictions always go to a human reviewer
- 1% random audit maintained permanently to catch model drift
- Camera field of view restricted to product surface only (no worker privacy risk)
- QC staff redeployed to model monitoring roles — not replaced
- Grad-CAM heatmaps used to explain why a unit was rejected

---

## Full Design Document

All 8 tasks are covered in detail in [`solution_report.md`](./solution_report.md):

1. Business domain choice and justification
2. Problem definition (stakeholders, current process, limitations)
3. AI task type identification with reasoning
4. Data requirement plan (types, collection, quality risks)
5. Model recommendation with alternatives considered
6. Evaluation plan (technical metrics + business KPIs)
7. Responsible AI risks and mitigations
8. One-page solution summary
