import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pkl")

def train():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} nahi mila!")
        return

    print("Dataset load ho raha hai...")
    df = pd.read_csv(CSV_PATH)
    print(f"Total samples: {len(df)}")
    
    # Pure NumPy contiguous arrays to avoid Arrow/Pandas indexing bug
    feature_cols = df.columns[:-1]
    label_col = df.columns[-1]

    X = np.ascontiguousarray(df[feature_cols].to_numpy(dtype=np.float32))
    y = np.array(df[label_col].astype(str).tolist())

    print(f"Feature count: {X.shape[1]}")
    print("Classes distribution:")
    print(pd.Series(y).value_counts())

    # Split dataset cleanly
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nModel training shuru ho rahi hai (Random Forest)...")
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n==========================================")
    print(f"Training Complete! Test Accuracy: {acc * 100:.2f}%")
    print(f"==========================================\n")
    print(classification_report(y_test, y_pred))

    # Save clean model
    joblib.dump(clf, MODEL_PATH)
    print(f"Clean model successfully saved: {MODEL_PATH}")

if __name__ == "__main__":
    train()