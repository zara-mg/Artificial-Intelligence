import os
import pandas as pd

# Try sensible locations for the dataset
possible_paths = [
    r"c:\Users\DELL\Documents\Artificial Intel\assessment 1\spam.xlsx",
    r"c:\Users\DELL\Documents\Artificial Intel\assessment 1\spam.csv",
    "spam.xlsx",
    "spam.csv"
]

dataset_path = next((p for p in possible_paths if os.path.exists(p)), None)

if dataset_path is None:
    # list files in the working directory and the script folder to help debugging
    cwd_files = os.listdir(os.getcwd())
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    script_files = os.listdir(script_dir)

    # try common filenames in script dir and cwd
    candidates = [
        os.path.join(script_dir, "spam.csv"),
        os.path.join(script_dir, "spam.xlsx"),
        os.path.join(os.getcwd(), "spam.csv"),
        os.path.join(os.getcwd(), "spam.xlsx"),
    ]

    for c in candidates:
        if os.path.exists(c):
            dataset_path = c
            break

    # try to find any csv/xlsx in the script directory
    if dataset_path is None:
        import glob
        for ext in ("*.csv", "*.xlsx"):
            matches = glob.glob(os.path.join(script_dir, ext))
            if matches:
                dataset_path = matches[0]
                break

    if dataset_path is None:
        # provide detailed guidance to the user and fail early
        raise FileNotFoundError(
            "Could not find spam dataset. Checked:\n"
            f"possible_paths: {possible_paths}\n"
            f"cwd files: {cwd_files}\n"
            f"script dir ({script_dir}) files: {script_files}\n"
            "Please place 'spam.csv' or 'spam.xlsx' in the script directory or update possible_paths."
        )


# Load depending on extension
if dataset_path.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(dataset_path)
else:
    df = pd.read_csv(dataset_path, encoding="latin-1")

# normalize expected columns
df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "message"})
df = df[["label", "message"]]
df["label"] = df["label"].astype(str).str.lower().map(lambda x: "spam" if "spam" in x else "ham")

print(df.head())


# --- Algorithm 1: Expert System ---
class ExpertSpamFilter:
    def __init__(self):
        self.rules = [
            ("free", "spam"),
            ("win", "spam"),
            ("buy now", "spam"),
            ("click", "spam"),
            ("limited offer", "spam"),
            ("urgent", "spam"),
            ("money", "spam"),
        ]

    def classify(self, message):
        text = str(message).lower()
        score = 0
        for keyword, label in self.rules:
            if keyword in text:
                score += 1
        return "spam" if score >= 2 else "ham"


# --- Algorithm 2: Fuzzy Logic Rule-Based System ---
class FuzzySpamFilter:
    def __init__(self):
        self.spam_words = ["free", "win", "offer", "buy", "click", "money", "urgent"]
        self.promotional_words = ["sale", "discount", "deal", "order now", "limited"]

    def fuzzy_score(self, message):
        text = str(message).lower()
        spam_hits = sum(word in text for word in self.spam_words)
        promo_hits = sum(word in text for word in self.promotional_words)

        # Fuzzy logic scoring
        spam_degree = spam_hits / len(self.spam_words)
        promo_degree = promo_hits / len(self.promotional_words)
        combined_score = (spam_degree * 0.7) + (promo_degree * 0.3)
        return combined_score

    def classify(self, message):
        score = self.fuzzy_score(message)
        if score > 0.6:
            return "spam"
        elif 0.3 <= score <= 0.6:
            return "likely spam"
        else:
            return "ham"


# Step 3: Create Instances of Both Algorithms
ai1 = ExpertSpamFilter()
ai2 = FuzzySpamFilter()

# Step 4: Apply Both Algorithms to the Dataset
df["AI1_Prediction"] = df["message"].apply(lambda x: ai1.classify(x))
df["AI2_Prediction"] = df["message"].apply(lambda x: ai2.classify(x))

# Step 5: Evaluate Accuracy
df["AI1_Correct"] = df["AI1_Prediction"] == df["label"]

# treat "likely spam" as spam for accuracy calculation
def ai2_correct(row):
    pred = row["AI2_Prediction"]
    label = row["label"]
    if pred == label:
        return True
    if pred == "likely spam" and label == "spam":
        return True
    return False

df["AI2_Correct"] = df.apply(ai2_correct, axis=1)

ai1_accuracy = df["AI1_Correct"].mean()
ai2_accuracy = df["AI2_Correct"].mean()

# Step 6: Print Accuracy Results
print(df[["label", "message", "AI1_Prediction", "AI2_Prediction"]].head(10))
print(f"\nExpertSpamFilter Accuracy: {ai1_accuracy * 100:.2f}%")
print(f"FuzzySpamFilter Accuracy: {ai2_accuracy * 100:.2f}%")
