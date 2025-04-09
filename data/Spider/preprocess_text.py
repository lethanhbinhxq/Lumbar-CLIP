import os
import pandas as pd
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))  
input_dir = os.path.join(script_dir, "raw_data")
overview_file = os.path.join(input_dir, "overview.csv")
grading_file = os.path.join(input_dir, "radiological_gradings.csv")
output_dir = os.path.join(script_dir, "preprocessed_text")
label_dir = os.path.join(script_dir, "label")

overview_df = pd.read_csv(overview_file)
grading_df = pd.read_csv(grading_file)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

def compress_ranges(numbers):
    if not numbers:
        return ""
    numbers = sorted(set(numbers))
    ranges = []
    start = prev = numbers[0]
    for num in numbers[1:]:
        if num == prev + 1:
            prev = num
        else:
            ranges.append((start, prev))
            start = prev = num
    ranges.append((start, prev))
    
    # Use en dash (U+2013) explicitly
    return ", ".join(f"{s}" if s == e else f"{s}-{e}" for s, e in ranges)

def summarize_patient_grading(patient_grading):
    pfirrmann_map = defaultdict(list)
    finding_map = defaultdict(list)

    for _, row in patient_grading.iterrows():
        disc = int(row["IVD label"])
        pfirrmann = int(row["Pfirrman grade"])
        pfirrmann_map[pfirrmann].append(disc)

        if row["Disc narrowing"]:
            finding_map["narrowing"].append(disc)
        if row["Disc bulging"]:
            finding_map["bulging"].append(disc)
        if row["LOW endplate"]:
            finding_map["low endplate"].append(disc)
        if row["UP endplate"]:
            finding_map["up endplate"].append(disc)
        if row["Disc herniation"]:
            finding_map["herniation"].append(disc)
        if row["Spondylolisthesis"]:
            finding_map["spondylolisthesis"].append(disc)
        if row["Modic"]:
            finding_map[f"Modic type {int(row['Modic'])}"].append(disc)

    parts = []

    for grade, discs in sorted(pfirrmann_map.items()):
        parts.append(f"Discs {compress_ranges(discs)} show Pfirrmann grade {grade}")

    for finding, discs in finding_map.items():
        parts.append(f"{finding.capitalize()} in discs {compress_ranges(discs)}")

    return ". ".join(parts) + "."

# Loop through each patient
label_0 = 0
label_1 = 0
for patient_id in grading_df["Patient"].unique():
    patient_grading = grading_df[grading_df["Patient"] == patient_id]
    caption = summarize_patient_grading(patient_grading)

    # Determine label
    lbp_values = patient_grading["LBP Label"]
    label = 0 if all(lbp_values == 0) else 1
    if label == 0:
        label_0 += 1
    else:
        label_1 += 1

    # Overview and file writing
    patient_overview = overview_df[overview_df["new_file_name"].str.startswith(f"{patient_id}_")]
    for _, row in patient_overview.iterrows():
        filename = row['new_file_name']
        if "t2_SPACE" in filename.lower():
            modality = "T2 SPACE-weighted MRI"
        elif "t2" in filename.lower():
            modality = "T2-weighted MRI"
        elif "t1" in filename.lower():
            modality = "T1-weighted MRI"
        else:
            modality = "Unknown MRI"

        sex = 'female' if row['sex'] == 'F' else 'male'
        text = f"A {modality} of a {sex} patient with {row['num_vertebrae']} vertebrae and {row['num_discs']} discs. "

        # Save caption
        out_file = os.path.join(output_dir, f"{filename}.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text + caption)

        # Save label
        label_file = os.path.join(label_dir, f"{filename}.txt")
        with open(label_file, "w", encoding="utf-8") as f:
            f.write(str(label))

        print(f"Processed {filename} → Caption and Label saved.")
    

print("Label 0: ", label_0)
print("Label_1: ", label_1)
