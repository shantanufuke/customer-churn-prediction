import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("churn_data.csv")

# Convert categorical data to numerical
df['tenure_bin'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60], labels=[1, 2, 3, 4, 5])

# Define features and labels
X = df[['tenure_bin', 'monthly_charges', 'total_charges']]
y = df['churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Model Evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Visualize Churn Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="churn", data=df)
plt.title("Churn Distribution")
plt.show()
