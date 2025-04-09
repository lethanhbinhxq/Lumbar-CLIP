import os
import pandas as pd

# Get base path (adjust if needed)
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "preprocessed_images")
text_dir = os.path.join(base_dir, "preprocessed_text")
label_dir = os.path.join(base_dir, "label")
output_csv = os.path.join(base_dir, "local_data", "lbp-meta.csv")

os.makedirs(os.path.join(base_dir, "local_data"), exist_ok=True)

rows = []
missing = 0

for img_filename in os.listdir(image_dir):
    if not img_filename.endswith(".png"):
        continue

    basename = os.path.splitext(img_filename)[0]
    image_path = os.path.join("preprocessed_images", img_filename)
    text_path = os.path.join(text_dir, f"{basename}.txt")
    label_path = os.path.join(label_dir, f"{basename}.txt")

    if not (os.path.exists(text_path) and os.path.exists(label_path)):
        print(f"Missing report or label for: {basename}")
        missing += 1
        continue

    with open(text_path, "r", encoding="utf-8") as f:
        report = f.read().strip()

    with open(label_path, "r", encoding="utf-8") as f:
        label = int(f.read().strip())

    rows.append({
        "imgpath": image_path,
        "report": report,
        "No Finding": 1 if label == 0 else 0,
        "LBP": label
    })

df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)

print(f"CSV created at: {output_csv}")
print(f"Total entries: {len(df)}")
print(f"Missing files skipped: {missing}")
print("Label counts:")
print(df["LBP"].value_counts())
