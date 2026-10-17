# HAI FA 26 Assignment 2: AI-Assisted Calorie Estimation
### Human-AI Complementarity in Visual Dietary Assessment with Nutrition5k

This repository contains the complete code, study dataset, model implementation, and evaluation pipeline for **HAI Assignment 2: Design an AI-Assisted Task**.

---

## Quick Replication Guide

To replicate the reported model evaluation metrics and error analysis breakdown on the study dataset:

```bash
pip install pandas numpy
python scripts/evaluate.py
```

This will read the study dataset (`data/study_dataset_predictions.csv`), compute:
- **Mean Absolute Error (MAE in kcal)** overall and across calorie tiers
- **Mean Absolute Percentage Error (MAPE)**
- **Pearson Correlation ($r$) & $R^2$**
- **Error Distribution Table** categorizing failure modes and assessing human-AI complementarity

---

## Repository Structure

```
HAI-Assignment-2/
├── data/
│   ├── raw_images/                         # 219 curated overhead meal photographs (Nutrition5k)
│   ├── study_dataset_ground_truth.csv       # Verified ground-truth nutrition & ingredient breakdown
│   ├── study_dataset_predictions.csv        # Primary study dataset (Gemini 3.5 Flash predictions & errors)
│   ├── study_dataset_predictions_QWEN.csv   # Open-source replication dataset (Qwen2-VL-2B predictions)
│   └── error_distribution_summary.csv      # Error analysis breakdown by failure category
├── scripts/
│   ├── download_nutrition5k.py             # Curates and streams balanced meal plates from GCS
│   ├── run_vlm_inference.py                # Runs multimodal Gemini 3.5 Flash VLM inference
│   ├── evaluate.py                         # Computes regression metrics and failure mode table
│   ├── compare_models.py                   # Multi-model benchmark & economic comparison script
│   └── colab_qwen2_vl_runner.ipynb         # Open-source Qwen2-VL-2B runner for Google Colab
└── README.md                               # Project documentation & replication guide
```

---

## Benchmark Results (219 Study Dataset Trials)

### 1. Primary Model Performance (Gemini 3.5 Flash, $0.86 Billed Cost)
| Metric | Measured Value | Description |
| :--- | :---: | :--- |
| **Evaluated Trials** | **219 / 219** | 100% completion with model consistency |
| **Mean Absolute Error (MAE)** | **161.1 kcal** | Average error against precision ground truth |
| **Median Absolute Error** | **115.1 kcal** | Robust error metric less sensitive to outliers |
| **Mean Absolute Pct Error (MAPE)**| **29.2%** | Relative percentage error across dishes |
| **Pearson Correlation ($r$)** | **0.760** ($R^2=0.578$) | Strong linear alignment with ground truth |

* **Low (100 to 350 kcal):** MAE 72.2 kcal (MAPE 36.1%)
* **Medium-Low (350 to 650 kcal):** MAE 95.5 kcal (MAPE 20.8%)
* **Medium-High (650 to 950 kcal):** MAE 205.3 kcal (MAPE 26.1%)
* **High (950 to 1600 kcal):** MAE 459.6 kcal (MAPE 41.8%)

### 2. Multi-Model Architectural and Economic Comparison
Run `python scripts/compare_models.py` to reproduce:

| Dimension | Primary Model (Gemini 3.5 Flash) | Open Replication (Qwen2-VL-2B) |
| :--- | :---: | :---: |
| **Architecture** | Frontier Proprietary Multimodal | 2.2B Open-Weights Multimodal |
| **Hosting Platform** | Google Cloud Serverless API | Google Colab Free T4 GPU |
| **Incurred Cost (219 Dishes)** | **$0.86 USD** (~$0.0039 / dish) | **$0.00 USD** (100% Free Compute) |
| **Mean Absolute Error (MAE)** | **161.1 kcal** | **267.3 kcal** |
| **Mean Abs Pct Error (MAPE)** | **29.2%** | **76.9%** |
| **Pearson Correlation ($r$)** | **0.760** ($R^2=0.578$) | **0.069** ($R^2=0.005$) |
| **Cognitive Behavior** | Scales with meal portion size | **Mean-anchoring bias** (~540 kcal flat) |


---

## Dataset Description

The study dataset consists of **219 task trials** curated from **Google's Nutrition5k** dataset (*Thames et al., CVPR 2021*), captured on an industrial RealSense scanner in Google cafeterias with precision gram scales.

Dishes are stratified across 4 balanced calorie tiers:
1. **Low (100 to 350 kcal):** 65 dishes (light salads, fruit bowls, egg white scrambles)
2. **Medium-Low (350 to 650 kcal):** 65 dishes (roasted chicken, vegetable and grain bowls)
3. **Medium-High (650 to 950 kcal):** 65 dishes (pasta, braised beef, hearty curry bowls)
4. **High (950 to 1600 kcal):** 24 dishes (combination platters, dense proteins and sides)

Columns in `data/study_dataset_predictions.csv`:
- `dish_id`: Unique identifier in Nutrition5k
- `tier`: Calorie quartile
- `image_path`: Path to overhead RGB image
- `ground_truth_calories`: Measured total kcal by nutritionists
- `total_mass_grams`: Measured weight in grams
- `total_fat_grams`, `total_carb_grams`, `total_protein_grams`: Macronutrient masses
- `num_ingredients`: Number of ingredients present
- `ingredient_summary`: Summary of primary food components
- `ingredient_breakdown_json`: Per-ingredient mass and calorie breakdown
- `model_predicted_calories`: Calorie estimate from Gemini 3.5 Flash
- `model_confidence`: Model self-reported confidence score (0.0 to 1.0)
- `model_reasoning`: Visual reasoning provided by the model
- `model_breakdown_json`: Model itemized ingredient & portion decomposition
- `error_kcal`: Absolute prediction error in kcal ($|\text{predicted} - \text{actual}|$)
- `percentage_error`: Absolute percentage error ($|\text{error}| / \text{actual} \times 100\%$)

---

## AI Model & Assistance Implementation

The AI assistant uses **Google Gemini 3.5 Flash** (`gemini-3.5-flash`), providing:
1. **Perceptual decomposition:** Itemized ingredient recognition.
2. **Portion gram estimation:** Estimated weight for each item.
3. **Baseline calorie total:** Initial reference anchor.
4. **Surface fat/dressing detection:** Identifies glazes and dressings that humans can calibrate.

To re-run inference across all 219 Nutrition5k plates:
```bash
pip install google-generativeai pillow pandas
python scripts/run_vlm_inference.py --model gemini-3.5-flash --delay 4.2
```

---

## Google Gen AI API Key Setup

Statistical evaluation on precomputed study results (`python scripts/evaluate.py` and `python scripts/compare_models.py`) does not require an API key or GPU compute.

To run fresh model inference or evaluate new plates using Gemini 3.5 Flash via `scripts/run_vlm_inference.py`, a Google Gen AI API key is required.

### Steps to Obtain an API Key:
1. Visit **Google AI Studio** at [https://aistudio.google.com/](https://aistudio.google.com/).
2. Sign in with a Google account.
3. Select **Get API key** in the navigation panel.
4. Click **Create API key** and choose an existing Google Cloud project or create a free-tier project.
5. Copy the generated API key.

### Configuring the API Key:
The inference script detects the key through any of the following approaches:

1. **Local `.env` File (Recommended):**  
   Create a `.env` file in the repository root directory:
   ```bash
   GEMINI_API_KEY=your_actual_api_key_here
   ```

2. **Environment Variable:**  
   Export the key in the current terminal session:
   - **Windows PowerShell:**
     ```powershell
     $env:GEMINI_API_KEY="your_actual_api_key_here"
     ```
   - **Windows Command Prompt:**
     ```cmd
     set GEMINI_API_KEY=your_actual_api_key_here
     ```
   - **Linux / macOS Bash:**
     ```bash
     export GEMINI_API_KEY="your_actual_api_key_here"
     ```

3. **Command-Line Argument:**  
   Pass the key explicitly when invoking the runner:
   ```bash
   python scripts/run_vlm_inference.py --api_key "your_actual_api_key_here"
   ```

If no key is configured beforehand, the script prompts for the key interactively upon launch.

To run a rapid test on a subset of 5 plates:
```bash
python scripts/run_vlm_inference.py --limit 5
```

