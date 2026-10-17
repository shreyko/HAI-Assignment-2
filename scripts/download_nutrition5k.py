import os
import csv
import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

METADATA_URLS = [
    "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv",
    "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe2.csv",
]
IMG_BASE_URL = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "raw_images")
OUTPUT_CSV = os.path.join(DATA_DIR, "study_dataset_ground_truth.csv")

TIER_TARGETS = {
    "Low": 65,
    "Medium-Low": 65,
    "Medium-High": 65,
    "High": 30,
}

TIERS = {
    "Low": (100.0, 350.0),
    "Medium-Low": (350.0, 650.0),
    "Medium-High": (650.0, 950.0),
    "High": (950.0, 1600.0),
}

def parse_metadata():
    dishes = []
    seen_ids = set()
    for url in METADATA_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                content = resp.read().decode("utf-8", errors="ignore")
        except Exception:
            continue

        reader = csv.reader(io.StringIO(content))
        for row in reader:
            if len(row) < 6:
                continue
            try:
                dish_id = row[0]
                if dish_id in seen_ids:
                    continue
                calories = float(row[1])
                mass = float(row[2])
                fat = float(row[3])
                carb = float(row[4])
                protein = float(row[5])
            
                ingredients = []
                num_ingrs = (len(row) - 6) // 7
                for j in range(num_ingrs):
                    base = 6 + j * 7
                    if base + 3 < len(row):
                        ingr_name = row[base + 1]
                        try:
                            ingr_grams = float(row[base + 2])
                            ingr_cals = float(row[base + 3])
                            ingredients.append({
                                "name": ingr_name,
                                "grams": round(ingr_grams, 1),
                                "calories": round(ingr_cals, 1)
                            })
                        except ValueError:
                            continue
                
                if mass >= 50.0 and len(ingredients) >= 2:
                    seen_ids.add(dish_id)
                    dishes.append({
                        "dish_id": dish_id,
                        "calories": round(calories, 1),
                        "mass_grams": round(mass, 1),
                        "fat_grams": round(fat, 1),
                        "carb_grams": round(carb, 1),
                        "protein_grams": round(protein, 1),
                        "num_ingredients": len(ingredients),
                        "ingredients": ingredients
                    })
            except ValueError:
                continue

    return dishes

def check_and_download_image(dish, tier_name):
    dish_id = dish["dish_id"]
    img_url = f"{IMG_BASE_URL}/{dish_id}/rgb.png"
    out_path = os.path.join(IMAGES_DIR, f"{dish_id}.png")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        dish_copy = dict(dish)
        dish_copy["tier"] = tier_name
        dish_copy["image_filename"] = f"{dish_id}.png"
        dish_copy["image_path"] = f"data/raw_images/{dish_id}.png"
        dish_copy["image_url"] = img_url
        return dish_copy

    try:
        req = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read()
            if len(data) > 5000:
                with open(out_path, "wb") as f:
                    f.write(data)
                dish_copy = dict(dish)
                dish_copy["tier"] = tier_name
                dish_copy["image_filename"] = f"{dish_id}.png"
                dish_copy["image_path"] = f"data/raw_images/{dish_id}.png"
                dish_copy["image_url"] = img_url
                return dish_copy
    except Exception:
        return None
    return None

def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    all_dishes = parse_metadata()
    curated_dishes = []
    
    for tier_name, (min_cal, max_cal) in TIERS.items():
        tier_target = TIER_TARGETS.get(tier_name, 55)
        candidates = [d for d in all_dishes if min_cal <= d["calories"] < max_cal]
        
        saved_count = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_dish = {
                executor.submit(check_and_download_image, d, tier_name): d 
                for d in candidates
            }
            for future in as_completed(future_to_dish):
                result = future.result()
                if result is not None:
                    curated_dishes.append(result)
                    saved_count += 1
                    if saved_count >= tier_target:
                        for f in future_to_dish:
                            f.cancel()
                        break

    fieldnames = [
        "dish_id",
        "tier",
        "ground_truth_calories",
        "total_mass_grams",
        "total_fat_grams",
        "total_carb_grams",
        "total_protein_grams",
        "num_ingredients",
        "ingredient_summary",
        "ingredient_breakdown_json",
        "image_filename",
        "image_path",
        "image_url"
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in curated_dishes:
            ingr_summary = ", ".join([f"{i['name']} ({i['grams']}g)" for i in d["ingredients"][:6]])
            if len(d["ingredients"]) > 6:
                ingr_summary += f", +{len(d['ingredients']) - 6} more"
            
            writer.writerow({
                "dish_id": d["dish_id"],
                "tier": d["tier"],
                "ground_truth_calories": d["calories"],
                "total_mass_grams": d["mass_grams"],
                "total_fat_grams": d["fat_grams"],
                "total_carb_grams": d["carb_grams"],
                "total_protein_grams": d["protein_grams"],
                "num_ingredients": d["num_ingredients"],
                "ingredient_summary": ingr_summary,
                "ingredient_breakdown_json": json.dumps(d["ingredients"]),
                "image_filename": d["image_filename"],
                "image_path": d["image_path"],
                "image_url": d["image_url"]
            })

if __name__ == "__main__":
    main()
