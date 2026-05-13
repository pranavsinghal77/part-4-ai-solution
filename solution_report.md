# Part 4: AI Solution Design — Manufacturing Visual Defect Inspection

**Name:** Pranav Singhal | **Student ID:** BA2511208 | **Course:** Certification in Business Analytics with Gen & Agentic AI

---

## Task 1: My Chosen Business Domain

I chose **Manufacturing** for this part of the project. My reason for this choice is that it connects directly to the image dataset I worked with in Part 2 (the defect classification problem), and it represents a genuinely high-impact area where AI can replace a slow, error-prone manual process with something measurable and scalable.

---

## Task 2: Defining the Business Problem

### What problem am I solving?
In high-volume manufacturing — think automotive components, electronics, or consumer goods — every finished product needs to pass a quality inspection before packaging. Right now, this is done by human inspectors who visually check each unit on a moving conveyor belt and manually flag anything that looks wrong.

The problem is that human inspection doesn't scale. At production speeds above 4–6 units per second, a human inspector physically cannot reliably assess every item. After 3–4 hours on shift, fatigue causes accuracy to drop by 15–20%. The result: roughly 7–8% of defective products escape to the customer, generating complaints, returns, and in severe cases, recalls.

### Who are the stakeholders?
| Stakeholder | Interest |
|---|---|
| QC Inspectors | Currently perform manual inspection; will shift to supervising the AI system |
| Production Line Managers | Responsible for throughput rates and defect escape rates |
| COO / Operations Director | Accountable for cost efficiency and customer satisfaction |
| End Customers | Directly impacted — they receive the final product |

### What does the current process look like?
A trained inspector stands at the end of the production line, visually examines each unit, and either lets it pass or removes it manually. Rejected items are logged on a paper sheet or spreadsheet. End-of-shift summaries are compiled for management. There is no real-time visibility into defect rates — a batch of bad products might not surface until the next morning's review.

### Why is the current process insufficient?
- **Fatigue and inconsistency:** No two inspectors classify borderline defects the same way, and accuracy drops sharply over a long shift
- **Speed bottleneck:** The line must slow down if inspection can't keep pace
- **No real-time analytics:** Paper logs can't trigger an immediate line adjustment when a pattern of defects appears
- **Cost:** Skilled QC staff are expensive; running inspection 24/7 requires multiple shifts of people

---

## Task 3: The AI Task Type I Identified

**Task Type: Image Classification**

I classified this as an image classification problem because each product image needs to be assigned to exactly one of four categories: `normal`, `scratch`, `dent`, or `stain`. The output is a single class label — not a bounding box around a region (that would be object detection), not a pixel-by-pixel map (that would be segmentation).

Image classification is appropriate here because the business decision is binary in practice: the product either passes or fails. The specific defect type is logged for process improvement analytics, but the primary action is just pass/reject. A CNN with a four-class softmax output handles this perfectly.

---

## Task 4: My Data Requirement Plan

### What data do I need?
My primary data source is images — specifically, high-resolution photos of product surfaces captured by an industrial camera mounted directly above the conveyor belt.

### Structured vs. unstructured
- **Unstructured (primary):** Product surface images in RGB format
- **Structured (secondary):** Labels (defect type, severity level), production metadata (batch ID, line ID, shift, timestamp, camera ID)

### Input features and target variable
| Element | Detail |
|---|---|
| Input | 224×224 RGB image of product surface |
| Target variable | `defect_class`: normal / scratch / dent / stain |
| Optional future target | `defect_severity`: minor / moderate / severe |

### How I would collect this data
1. **Retroactive labelling:** Pull 6 months of archived camera footage from the production line. Use a labelling tool (Label Studio or Scale AI) with two annotators per image to build the initial training corpus. Target: at least 500 images per class to start, 2,000+ per class for production quality.
2. **Active learning loop:** Once the model is deployed, I'd surface low-confidence predictions (where the model is uncertain) and route them to human experts for labelling — this continuously grows the dataset in the most useful areas.
3. **Augmentation:** For classes with fewer examples (e.g., rare defect types), I'd apply flipping, rotation, brightness jitter, and zoom to synthetically expand the training set.

### Data quality risks I need to manage
| Risk | How I'd handle it |
|---|---|
| Label noise (annotators disagree) | Require inter-annotator agreement ≥ 0.90; flag disagreements for senior review |
| Class imbalance (rare defects) | Targeted data collection campaigns + augmentation |
| Lighting drift between shifts | Capture camera metadata; include day/night conditions in training data |
| New product variants not in training set | Collect representative samples before deploying to a new product line |

---

## Task 5: My Model Recommendation

**Recommended model: CNN with Transfer Learning — EfficientNet-B0 (ImageNet pre-trained)**

### Why a CNN?
CNNs are the established standard for image classification because they learn hierarchical visual features directly from pixel data. Early layers detect edges and gradients; middle layers detect shapes and textures; deep layers detect semantically meaningful patterns like dents or scratches. The key advantage over a dense network is that CNN filters are spatially shared — a 3×3 filter learns one visual pattern and applies it everywhere in the image, making them far more parameter-efficient.

### Why transfer learning?
Training a CNN from scratch on 2,000 images is risky — the model may overfit. EfficientNet-B0 was pre-trained on ImageNet (1.2 million images, 1,000 classes) and has already learned universal visual features: edges, textures, shapes. I only need to fine-tune the final classification layer on my four defect classes. This achieves strong performance even with a relatively small labelled dataset, which is realistic in an industrial setting where defect data takes time to collect.

### Deployment architecture I'd use
```
Camera → Frame capture → Preprocessing (resize 224×224, normalise)
  → EfficientNet-B0 (TF SavedModel on NVIDIA Jetson edge device)
  → Defect class + confidence score
        ├── Confidence ≥ 0.85 → Automated PLC reject/pass signal
        └── Confidence < 0.85 → Human review queue
  → All predictions logged → Dashboard + Data Warehouse
```

### Why not other options?
| Option | My reasoning |
|---|---|
| Custom CNN from scratch | Fine for prototyping (as in Part 2), but inferior baseline with limited real-world data |
| ResNet-50 | Higher accuracy ceiling, but 4× more parameters → slower edge inference |
| Vision Transformer (ViT) | Best accuracy at scale; overkill for 224×224 images in a latency-sensitive edge deployment |
| YOLO (object detection) | Over-engineered — I don't need to draw bounding boxes around the defect, just classify the unit |

---

## Task 6: My Evaluation Plan

### Technical metrics I'd use
| Metric | My target | Why it matters |
|---|---|---|
| Recall (defect classes) | > 98% | Missing a defect is far more costly than a false alarm |
| Precision (defect classes) | > 90% | Avoid rejecting too many good products (costly waste) |
| F1-Score (Macro) | > 93% | Balanced measure across all 4 classes |
| Overall accuracy | > 95% | General sanity check |
| AUC-ROC | > 0.97 | Threshold-independent quality measure |
| Inference latency | < 50ms | Required to keep pace with line speed |

### Business KPIs I'd track
| KPI | Baseline | 12-month target |
|---|---|---|
| Defect escape rate | ~7–8% | < 1% |
| Customer defect complaints | Month 1 baseline | −60% |
| QC labour cost | 100% | −40% |
| Inspection throughput | 4 items/sec | 20+ items/sec |
| Cost per defect detected | INR 180 | INR 12 |

### Failure cases I'd plan for
1. A novel defect type not seen in training — the model would assign it to the closest known class with high (false) confidence. I'd catch this with a monthly human audit on a random sample.
2. Camera lens gets dirty or the lighting changes — confidence scores would drop across the board. I'd monitor the mean confidence distribution as an early warning signal.
3. A new product variant gets introduced — the model sees something it was never trained on and misclassifies it. Standard procedure: collect 200+ images of the new variant before switching the line over.

### Human review process
- **Months 0–3:** Every AI rejection reviewed by a human before physical removal
- **Months 3–6:** Only predictions with confidence < 0.85 go to human review
- **Month 6+:** 1% random sampling of all predictions, weekly calibration report comparing AI labels to human audit labels, automatic retraining triggered if accuracy drops > 3%

---

## Task 7: Responsible AI Considerations

When I think about deploying this in a real factory, there are several risks that go beyond model accuracy.

**Bias in training data:** If I collected all my training images from one factory, one camera, and one lighting setup, the model will be biased toward those specific conditions. It might fail completely at a different plant. My mitigation: deliberately collect training data from multiple sites and shifts before declaring the model production-ready.

**Incorrect predictions and their consequences:** A false negative (defective product classified as normal) is dangerous — it reaches the customer. A false positive (good product rejected) wastes material and triggers unnecessary line stoppages. I addressed this by prioritising Recall over Precision in my evaluation plan, and by keeping humans in the loop for low-confidence cases.

**Privacy of workers:** Industrial cameras capture more than just products — they may capture workers' hands, faces, or movements. I would restrict the camera field of view to the product surface only, encrypt all stored image data, and enforce a 30-day data retention policy.

**Over-reliance on the system:** The risk I find most serious is that the production team starts treating the AI as infallible and disables the human audit process entirely. My policy recommendation: maintain a mandatory 1% random human audit permanently, regardless of how well the model performs. The AI is a decision-support tool, not the final authority.

**Impact on workers:** QC inspectors may fear job losses when this system is introduced. My position is that redeployment — not replacement — is the right approach. Inspectors become model monitors, exception handlers, and data labellers. Communicating this early, before deployment, is essential for gaining operator trust.

**Explainability:** If a batch gets flagged and rejected, a production manager needs to understand why. I'd implement Grad-CAM heatmaps that highlight which part of the image triggered the classification — showing "the model rejected this because of this region near the edge" builds trust and enables faster investigation.

---

## Task 8: My Final One-Page Solution Summary

### The Problem
Manual visual inspection on manufacturing production lines misses 7–8% of surface defects due to inspector fatigue, speed limitations, and inconsistent classification standards. Defective products reach customers, generating complaints and costly returns.

### My Proposed AI Solution
Deploy a CNN-based image classification system using EfficientNet-B0 (fine-tuned on labelled defect images) mounted on an edge device at the end of the production line. An industrial camera captures each product as it passes. The model classifies it in under 50ms as `normal`, `scratch`, `dent`, or `stain`. Units classified as defective with high confidence are automatically rejected via a PLC signal. Low-confidence cases go to a human reviewer.

### Data I Would Need
- 2,000–5,000 labelled product images per defect class
- Captured under real production line conditions (varying lighting, angles, speeds)
- Labelled by trained annotators with inter-annotator agreement ≥ 0.90

### Why This Model
EfficientNet-B0 with ImageNet transfer learning achieves strong performance with limited labelled data — realistic in an industrial setting. It runs in under 50ms on an edge device, meeting the throughput requirement. It can be incrementally fine-tuned as new defect types are introduced.

### Expected Business Impact
| Metric | Before | After |
|---|---|---|
| Defect escape rate | ~7–8% | < 1% |
| Throughput | 4 items/sec | 20+ items/sec |
| QC labour cost | Baseline | −40% |
| Customer complaints | Baseline | −60% |

### My Risk and Mitigation Plan
| Risk | Mitigation |
|---|---|
| Novel defect types missed | Monthly random human audit; active learning pipeline |
| Camera/lighting drift | Monitor confidence distribution; trigger retraining at >3% accuracy drop |
| Worker displacement fears | Redeploy to monitoring roles; communicate this before launch |
| Over-reliance on AI | Mandatory 1% random audit maintained permanently |
| Worker privacy | Camera restricted to product surface; 30-day image retention policy |
