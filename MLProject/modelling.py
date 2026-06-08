import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Kita HAPUS mlflow.set_tracking_uri() agar MLflow menyimpan log-nya langsung ke folder lokal GitHub.
mlflow.set_experiment("Telco_Churn_CI_Experiment")
mlflow.sklearn.autolog()

# Mengambil data yang ada di folder yang sama (di MLProject)
df = pd.read_csv("telco_churn_clean.csv")

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Memulai pelatihan model di GitHub Actions...")
with mlflow.start_run(run_name="CI_Run"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model berhasil dilatih dengan akurasi: {acc:.4f}")