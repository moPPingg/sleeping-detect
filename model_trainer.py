"""
TRAIN MODEL - SCRIPT DON GIAN
"""
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("="*80)
print("TRAIN MODEL - DRIVER MONITORING SYSTEM")
print("="*80)

# Load data
print("\n[1] Dang load du lieu...")
df = pd.read_csv('face_data.csv')
print(f"[OK] Da load: {df.shape[0]} mau, {df.shape[1]} features")

# Check labels
print("\n[2] Kiem tra phan phoi du lieu:")
label_names = {0: 'Tinh Tao', 1: 'Buon Ngu', 2: 'Cui Xuong', 3: 'Ngu Gat'}
for label in sorted(df['label'].unique()):
    count = len(df[df['label'] == label])
    print(f"  Label {label} ({label_names.get(label, 'Unknown')}): {count} mau ({count/len(df)*100:.1f}%)")

# Prepare data
print("\n[3] Chuan bi du lieu...")
X = df.drop('label', axis=1).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[OK] Training set: {X_train.shape[0]} mau")
print(f"[OK] Test set: {X_test.shape[0]} mau")

# Train model
print("\n[4] Dang train Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("[OK] Da train xong!")

# Evaluate
print("\n[5] Danh gia model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*80}")
print(f"ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*80}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                            target_names=[label_names[i] for i in range(4)]))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print("          Predicted")
print("         ", "  ".join([f"{i:3}" for i in range(4)]))
for i, row in enumerate(cm):
    if i == 0:
        print(f"Actual {i} ", end="")
    else:
        print(f"       {i} ", end="")
    print("  ".join([f"{val:3}" for val in row]))

# Save model
print("\n[6] Luu model...")
with open('drowsiness_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("[OK] Da luu model vao: drowsiness_model.pkl")

print("\n" + "="*80)
print("HOAN THANH! Ban co the chay: python run_advanced_dmsNEW.py")
print("="*80)

