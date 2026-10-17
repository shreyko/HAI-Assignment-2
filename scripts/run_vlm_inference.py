import os
import sys
import json
import time
import argparse
import pandas as pd
from PIL import Image

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai is not installed. Run: pip install google-generativeai")
    sys.exit(1)

def load_dotenv_simple():
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

load_dotenv_simple()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INPUT_CSV = os.path.join(DATA_DIR, "study_dataset_ground_truth.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "study_dataset_predictions.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "raw_images")

VLM_PROMPT = """You are an expert nutritionist assisting a human task performer in a visual dietary assessment study.
Analyze this overhead photograph of a plated meal.

Task:
1. Identify the visible food items and ingredients present on the plate.
2. Estimate the portion weight (in grams) and calorie count (in kcal) for each item.
3. Note any visible cooking oils, butter glazes, dressings, or sauces.
4. Provide your overall estimated total calories (kcal) for the entire plate.
5. Provide a confidence score between 0.0 and 1.0.
6. Provide a concise 1-2 sentence visual reasoning for your estimate.

Respond strictly in valid JSON format with no markdown fencing, using this exact schema:
{
  "breakdown": [
    {"item": "grilled chicken breast", "estimated_grams": 160, "calories": 260},
    {"item": "white rice", "estimated_grams": 180, "calories": 235}
  ],
  "total_calories": 495,
  "confidence_score": 0.85,
  "reasoning": "Standard restaurant portion of grilled chicken served over white rice with light seasoning."
}
"""

def save_df_safe(df, path):
    for attempt in range(5):
        try:
            df.to_csv(path, index=False)
            return
        except PermissionError:
            time.sleep(0.5)
    temp_path = path + ".tmp"
    df.to_csv(temp_path, index=False)
    try:
        os.replace(temp_path, path)
    except Exception:
        pass

def parse_json_response(raw_text):
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
        total_cals = float(data.get("total_calories", 0))
        confidence = float(data.get("confidence_score", 0.5))
        reasoning = str(data.get("reasoning", ""))
        breakdown = data.get("breakdown", [])
        return total_cals, confidence, reasoning, json.dumps(breakdown)
    except Exception as e:
        import re
        match = re.search(r'"total_calories"\s*:\s*([0-9.]+)', text)
        if match:
            return float(match.group(1)), 0.5, "Parsed via regex fallback", "[]"
        return None, None, f"JSON parse error: {e}", "[]"

def run_inference(api_key, model_name="gemini-3.5-flash", limit=None, delay=4.2, reset=False):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name)

    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found! Run download_nutrition5k.py first.", flush=True)
        return

    if os.path.exists(OUTPUT_CSV) and not reset:
        df = pd.read_csv(OUTPUT_CSV)
    else:
        df = pd.read_csv(INPUT_CSV)
        df["model_predicted_calories"] = None
        df["model_confidence"] = None
        df["model_reasoning"] = None
        df["model_breakdown_json"] = None
        df["error_kcal"] = None
        df["percentage_error"] = None
        save_df_safe(df, OUTPUT_CSV)

    pending_mask = df["model_predicted_calories"].isna()
    pending_count = pending_mask.sum()
    completed_already = len(df) - pending_count
    print(f"Model: {model_name} | Dishes pending: {pending_count} (already completed: {completed_already}/{len(df)})", flush=True)

    if pending_count == 0:
        print("All dishes already have predictions!", flush=True)
        return df

    processed = 0
    for idx, row in df.iterrows():
        if pd.notna(row["model_predicted_calories"]):
            continue

        dish_id = row["dish_id"]
        tier = row["tier"]
        actual_cals = float(row["ground_truth_calories"])
        img_path = os.path.join(IMAGES_DIR, f"{dish_id}.png")

        if not os.path.exists(img_path):
            print(f"  [Warning] Missing image: {img_path}", flush=True)
            continue

        success = False
        for attempt in range(4):
            try:
                pil_img = Image.open(img_path)
                response = model.generate_content([VLM_PROMPT, pil_img])
                pred_cals, confidence, reasoning, breakdown_json = parse_json_response(response.text)

                if pred_cals is not None and pred_cals > 0:
                    err_kcal = abs(pred_cals - actual_cals)
                    pct_err = (err_kcal / actual_cals) * 100.0

                    df.at[idx, "model_predicted_calories"] = round(pred_cals, 1)
                    df.at[idx, "model_confidence"] = round(confidence, 2) if confidence else None
                    df.at[idx, "model_reasoning"] = reasoning
                    df.at[idx, "model_breakdown_json"] = breakdown_json
                    df.at[idx, "error_kcal"] = round(err_kcal, 1)
                    df.at[idx, "percentage_error"] = round(pct_err, 1)

                    processed += 1
                    total_done = completed_already + processed
                    print(f"[{total_done}/{len(df)}] {dish_id} ({tier}) -> Actual: {actual_cals:.0f} | Pred: {pred_cals:.0f} kcal | Error: {err_kcal:.0f} kcal ({pct_err:.1f}%)", flush=True)
                    save_df_safe(df, OUTPUT_CSV)
                    success = True
                    break
                else:
                    time.sleep(2)

            except Exception as e:
                err_str = str(e).lower()
                if "quota" in err_str or "rate" in err_str or "429" in err_str:
                    wait_s = 20 * (attempt + 1)
                    time.sleep(wait_s)
                else:
                    time.sleep(3)

        if not success:
            print(f"  [Failed] Could not get prediction for {dish_id} after retries.", flush=True)

        if limit and processed >= limit:
            break

        time.sleep(delay)

    save_df_safe(df, OUTPUT_CSV)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM inference on Nutrition5k dataset")
    parser.add_argument("--api_key", type=str, default=os.environ.get("GEMINI_API_KEY", ""), help="Google Gemini API key")
    parser.add_argument("--model", type=str, default="gemini-3.5-flash", help="Model name (default: gemini-3.5-flash)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of dishes to process")
    parser.add_argument("--delay", type=float, default=4.2, help="Delay between API requests in seconds (default: 4.2)")
    parser.add_argument("--reset", action="store_true", help="Reset all predictions and run fresh from start")

    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        api_key = input("Enter your Gemini API Key: ").strip()

    if not api_key:
        print("Error: API key is required!", flush=True)
        sys.exit(1)

    run_inference(api_key=api_key, model_name=args.model, limit=args.limit, delay=args.delay, reset=args.reset)
