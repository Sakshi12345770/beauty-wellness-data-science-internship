# Beauty & Wellness - Churn Prediction & Recommendation
# Intern: Sakshi Wankhade

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample Data Creation (replace with Kaggle dataset in Week 2)
data = {
    'CustomerID': [1,2,3,4,5,6,7,8,9,10],
    'Recency': [5, 60, 10, 80, 15, 90, 20, 45, 30, 100], # days since last purchase
    'Frequency': [10, 1, 8, 1, 5, 0, 6, 2, 4, 1],
    'Monetary': [5000, 500, 4000, 300, 2500, 100, 3000, 800, 1500, 200],
    'Avg_Rating': [4.5, 2.0, 4.0, 1.5, 3.5, 1.0, 4.2, 2.5, 3.0, 1.2],
    'Churn': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] # 0=Retain, 1=Churn
}
df = pd.DataFrame(data)

print("Dataset Head:")
print(df.head())

# EDA - Churn Distribution
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.savefig('churn_plot.png')
plt.show()

# Model Training for Churn Prediction
X = df[['Recency', 'Frequency', 'Monetary', 'Avg_Rating']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Feature Importance - Why customer churns
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(importances)

# Recommendation Logic (Simple Rule-based)
def recommend_product(skin_type):
    if skin_type == "Oily":
        return "Salicylic Acid Cleanser + Niacinamide Serum"
    elif skin_type == "Dry":
        return "Hyaluronic Acid Moisturizer + Ceramide Cream"
    else:
        return "Vitamin C Serum + Gentle Cleanser"

print("\nRecommendation for Oily Skin:", recommend_product("Oily"))
