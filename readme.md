

# Pneumonia Detection from Chest X-rays : ResNet-50 (PyTorch)

website link : https://cxr-frontend-2vlpq962i-arkins-projects-d42a3382.vercel.app/

> **Purpose.** Build and evaluate a binary classifier (PNEUMONIA vs NORMAL) on the Kaggle Chest X-ray dataset, with GPU-accelerated training, transparent evaluation, and deployment notes for a clinical-assistive workflow.

---

## 1) Dataset & Splits

* **Source:** Kaggle Chest X-ray (Pneumonia).
* **Final split (after cleanup/re-split):**

  * Train: **4,434** images (NORMAL: **1,140**, PNEUMONIA: **3,294**)
  * Val: **782** images
  * Test: **624** images
* **Leakage control:** Validation created from train with stratified (patient-aware where detectable) sampling. Test kept **as provided** by dataset.

---

## 2) Preprocessing

* **Resize:** `224×224`
* **Normalization:** **ImageNet mean/std**
* **Augmentations (train only):** `RandomResizedCrop`, small rotation; **horizontal flip disabled** (`left/right` matters anatomically in CXR).
* **Rationale:** Light geometric jitter improves generalization; flips remain off to preserve side-specific findings.

---

## 3) Model Architecture

* **Backbone:** **ResNet-50** pretrained on ImageNet.
* **Head:** Replace `fc` with **single logit** → `BCEWithLogitsLoss`.
* **Imbalance handling:** `pos_weight = 0.346` (neg/pos ratio) inside BCE.
* **Params:** ~**23.51 M**.

---

## 4) Training Setup

* **Hardware:** RTX 4060, mixed precision (**AMP**), TF32 enabled, cuDNN autotune.
* **Optimizer / Schedule:** **AdamW** + **OneCycleLR**.
* **Max epochs requested:** 60
* **Early stopping:** on **val AUC** — **best epoch = 6**.
* **Checkpointing:** Best-AUC weights saved; TorchScript export produced for demo/inference.

---

## 5) Evaluation Strategy

1. Train on **train**, monitor on **val** with AUC/F1/Recall/Specificity.
2. **Pick decision threshold** by maximizing **F1 on validation**.
3. **Freeze that threshold** and report **test** metrics.
4. Plot **ROC**, **PR**, **Calibration (reliability)**, and **Confusion matrices**.
5. Discuss **operating points** vs intended clinical role (triage vs flagging).

---

## 6) Results

### 6.1 Validation (threshold chosen by best-F1: **τ = 0.05**)

* **AUC:** **0.998**
* **AP (PR-AUC):** **0.999**
* **Accuracy:** 0.977
* **F1:** 0.985
* **Precision:** 0.980
* **Recall (Sensitivity):** 0.990
* **Specificity:** 0.940
* **Confusion (Val):** TP=575, TN=189, FP=12, FN=6

### 6.2 Test (evaluated at same **τ = 0.05**)

* **AUC:** **0.981**
* **AP (PR-AUC):** **0.984**
* **Accuracy:** 0.788
* **F1:** 0.855
* **Precision:** 0.748
* **Recall (Sensitivity):** **0.997** (very high)
* **Specificity:** **0.440** (low)
* **Confusion (Test):** TP=389, TN=103, FP=131, FN=1

### 6.3 Probability Quality (Calibration)

* **Brier (Val/Test):** 0.025 / 0.063 → reasonable but can improve with temperature scaling or Platt.

> **Takeaway:** At τ=0.05, the model aggressively favors **recall** (catching nearly every pneumonia) at the expense of **specificity** (many false positives) on the test set. AUC remains strong (0.98), so shifting the threshold upward should recover specificity substantially with a controlled recall trade-off.

---

## 7) Operating Points & Thresholding

Pick τ based on your **intended use**:

* **Triage (high-recall)** — Keep **low τ** to **prioritize sensitivity**; accept more FPs to avoid missing cases (current τ=0.05 is an example).
* **Balanced review** — Set τ via **Youden’s J** (maximize `TPR−FPR` on val ROC) or maximize **F1** but verify on **test** that specificity is acceptable.
* **Precision-oriented flagging** — Raise τ (e.g., sweep τ∈[0.2, 0.6] on val), pick τ with **target specificity** (e.g., ≥0.85), then report test metrics at that τ.

> **How to adjust:** in code, sweep τ on **validation**, choose τ for your target (**recall**, **precision**, or **specificity**), and **lock** that τ for test and deployment.

---

## 8) Error Analysis (qualitative guide)

* **Common FPs:** Devices/tubes, projection/positioning differences (AP vs PA), under/overexposure, non-pneumonia opacities.
* **Rare FNs:** Very subtle early-stage opacities or atypical presentations.
* **Next steps:** Review **most confident mistakes**; add targeted augmentation (exposure/contrast jitter), increase input size (256–288), or domain-adapt.

---

## 9) Real-World Deployment Considerations

> **This model is for research only. Not for clinical use without regulatory clearance.**

* **Generalizability / Domain shift**

  * Validate across **multiple sites**, devices, age groups, and AP/PA views.
  * Monitor for drift; recalibrate threshold and probabilities over time.
* **Clinical workflow**

  * **Assistive triage**: route high-risk studies for **priority** radiologist review.
  * Integrate with PACS/RIS; store inference + heatmaps; ensure human override.
* **Threshold policy**

  * Choose τ aligned to clinical priorities (e.g., **high sensitivity** in ED triage).
  * Document policy in SOP; revisit periodically.
* **Risk management**

  * **False negatives**: delayed care; add safety nets (e.g., second reader).
  * **False positives**: workload; mitigate with batching, secondary checks.
* **Governance & privacy**

  * PHI handling, audit logs, access control, versioning, rollbacks.
  * Fairness across demographics; subgroup performance checks.
* **Regulation**

  * Treat as **SaMD** if used clinically: prospective validation, post-market surveillance.

---

## 10) Reproducibility & Assets

* **Training early-stop epoch:** **6** (best val AUC)
* **Figures:**

  * ROC: `runs_resnet50/report_assets/roc.png`
  * Precision–Recall: `runs_resnet50/report_assets/pr.png`
  * Calibration: `runs_resnet50/report_assets/calibration.png`
  * Confusion (Val): `runs_resnet50/report_assets/cm_val.png`
  * Confusion (Test): `runs_resnet50/report_assets/cm_test.png`
  * Learning curves: `runs_resnet50/report_assets/learning_curves.png`
* **Serialized artifacts:**

  * TorchScript: `runs_resnet50/resnet50_pneumonia_traced.pt`
  * JSON: `runs_resnet50/best_val.json`, `runs_resnet50/test_metrics.json`, `runs_resnet50/train_history.json`, `runs_resnet50/report_summary.json`

---

## 11) Recommended Next Experiments

1. **Threshold retune** for a **balanced** or **precision-oriented** operating point; re-report test metrics.
2. **Calibration**: add **temperature scaling** on val; re-plot reliability.
3. **Input resolution**: try **256–288** (with AMP, should fit 4060 VRAM).
4. **Augmentations**: mild contrast/brightness jitter; avoid HFlip unless justified.
5. **Model variants**: EfficientNet-B2/ConvNeXt-Tiny; or light ensembling.
6. **Site/domain adaptation**: fine-tune on target-site data if available.

---

