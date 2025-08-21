import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv("covtype.csv")

x = df.drop('Cover_Type', axis=1)
y = df['Cover_Type']

y = y - 1

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

xg_model = XGBClassifier(n_estimators=500,learning_rate=0.3,max_depth=7)
xg_model.fit(x_train, y_train)

y_pred = xg_model.predict(x_test)

# Check the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# The confusion matrix shows us which classes the model is confusing
print("\nGenerating confusion matrix...")
ConfusionMatrixDisplay.from_estimator(xg_model, x_test, y_test, xticks_rotation='vertical')
plt.title('XGBoost Confusion Matrix')
plt.show()

importances = xg_model.feature_importances_

feature_importances = pd.Series(importances, index=x.columns)

top_n = 20
top_features = feature_importances.nlargest(top_n).sort_values(ascending=True)

plt.figure(figsize=(12, 8))
top_features.plot(kind='barh', color='lightgreen')
plt.title(f'Top {top_n} Most Important Features (XGBoost)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()

plt.show()
