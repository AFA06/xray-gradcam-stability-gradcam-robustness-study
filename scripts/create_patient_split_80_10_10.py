import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load full 30k label CSV
csv_path = "data/labels/train_30k_final.csv"
df = pd.read_csv(csv_path)

print("Total images:", len(df))

# Extract patient ID from filename
df["PatientID"] = df["Path"].apply(lambda x: x.split("_")[0])

# Get unique patients
patients = df["PatientID"].unique()
print("Total unique patients:", len(patients))

# First split: 80% train, 20% temp
train_patients, temp_patients = train_test_split(
    patients, test_size=0.2, random_state=42, shuffle=True
)

# Second split: 10% val, 10% test (split 20% temp equally)
val_patients, test_patients = train_test_split(
    temp_patients, test_size=0.5, random_state=42, shuffle=True
)

# Create splits
train_df = df[df["PatientID"].isin(train_patients)]
val_df = df[df["PatientID"].isin(val_patients)]
test_df = df[df["PatientID"].isin(test_patients)]

print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("Test size:", len(test_df))

# Save
train_df.to_csv("data/labels/train_split.csv", index=False)
val_df.to_csv("data/labels/val_split.csv", index=False)
test_df.to_csv("data/labels/test_split.csv", index=False)

print("Splits saved successfully.")
