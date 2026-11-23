import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

print("=" * 60)
print("WEEK 8 DEMO (FINAL): Training Hypertension Risk Model")
print("=" * 60)

np.random.seed(42)
n = 1000

age = np.random.randint(20, 80, n)
bmi = np.random.uniform(18, 40, n)
smoking = np.random.binomial(1, 0.3, n)
alcohol = np.random.binomial(1, 0.4, n)
physical_activity = np.random.binomial(1, 0.6, n)
family_history = np.random.binomial(1, 0.25, n)

risk_score = (-5.0 + 0.05*age + 0.12*bmi + 0.8*smoking + 0.3*alcohol
              - 0.7*physical_activity + 1.0*family_history)
prob = 1 / (1 + np.exp(-risk_score))
hypertension = (np.random.random(n) < prob).astype(int)

df = pd.DataFrame({'age':age,'bmi':bmi,'smoking':smoking,'alcohol':alcohol,
                   'physical_activity':physical_activity,
                   'family_history':family_history,'hypertension':hypertension})

X, y = df.drop('hypertension', axis=1), df['hypertension']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
print(f"✓ Accuracy: {acc*100:.2f}%")

with open('model_week8.pkl', 'wb') as f:
    pickle.dump(model, f)
X.sample(100, random_state=42).to_csv('background_week8.csv', index=False)
with open('feature_names_week8.pkl', 'wb') as f:
    pickle.dump(['age','bmi','smoking','alcohol','physical_activity','family_history'], f)
print("✅ WEEK 8 MODEL TRAINING COMPLETE!")