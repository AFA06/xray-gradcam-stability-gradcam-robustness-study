import pandas as pd
import re
from sklearn.model_selection import train_test_split

CSV_IN = "data/labels/train_30k_final.csv"
TRAIN_OUT = "data/labels/train_30k_patient_train.csv"
VALID_OUT = "data/labels/train_30k_patient_valid.csv"

RANDOM_SEED = 42
VAL_RATIO = 0.2

print("Loading CSV...")
df = pd.read_csv(CSV_IN)

# -------------------------
# Extract patient ID
# -------------------------
def extract_patient(path):
    m = re.match(r"(patient\d+)_", str(path))
    if m is None:
        raise ValueError(f"Could not extract patient from: {path}")
    return m.group(1)

df["patient_id"] = df["Path"].apply(extract_patient)

# -------------------------
# Split by PATIENT
# -------------------------
patients = df["patient_id"].unique()
print(f"Total unique patients: {len(patients)}")

train_p, val_p = train_test_split(
    patients,
    test_size=VAL_RATIO,
    random_state=RANDOM_SEED,
    shuffle=True
)

train_df = df[df["patient_id"].isin(train_p)].drop(columns=["patient_id"])
val_df   = df[df["patient_id"].isin(val_p)].drop(columns=["patient_id"])

print(f"Train images: {len(train_df)}")
print(f"Valid images: {len(val_df)}")

# -------------------------
# Save
# -------------------------
train_df.to_csv(TRAIN_OUT, index=False)
val_df.to_csv(VALID_OUT, index=False)

print("Saved:")
print(" ", TRAIN_OUT)
print(" ", VALID_OUT)
