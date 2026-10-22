"""
Data Sourcing & Initial Analysis Planning
Intern: Sakshi Wankhade
"""

import pandas as pd
import numpy as np
import kagglehub
import os

print("=== WEEK 2 DATA SOURCING STRATEGY ===")

# Step 1: Download Datasets (Kaggle API)
# Note: For GitHub, we create sample merged data if kaggle key not available

# Creating Sample Master Dataset by merging Sephora + Customer Personality logic
np.random.seed(42)
n = 1000

# Part A: Customer Personality Data (for Churn)
customers = pd.DataFrame({
    'CustomerID': range(1, n+1),
    'Age': np.random.randint(18, 55, n),
    'Income': np.random.randint(20000, 100000, n),
    'Recency': np.random.randint(1, 120, n),
    'Frequency': np.random.randint(1, 20, n),
    'Monetary': np.random.randint(500, 15000, n),
    'Wishlist_Count': np.random.randint(0, 25, n),
    'Complaints': np.random.randint(0, 3, n)
})

# Part B: Skin Profile (from Sephora dataset logic)
customers['Skin_Type'] = np.random.choice(['Oily','Dry','Combination','Sensitive'], n, p=[0.4,0.2,0.3,0.1])
customers['Skin_Concern'] = np.random.choice(['Acne','Pigmentation','Dryness','Ageing','Dullness'], n)

# Part C: Create Churn Label based on Week 1 Hypothesis
customers['Churn'] = ((customers['Recency'] > 45) & (customers['Wishlist_Count'] < 5)).astype(int)
customers.loc[np.random.choice(customers.index, 100), 'Churn'] = 1  # noise

# Part D: Reviews Data (from Sephora)
reviews = []
for cust_id in customers['CustomerID'].sample(800):
    reviews.append({
        'CustomerID': cust_id,
        'ProductID': np.random.randint(100, 500),
        'Rating': np.random.choice([1,2,3,4,5], p=[0.1,0.1,0.2,0.3,0.3]),
        'Review_Text': np.random.choice([
            'Great for oily skin, reduced acne',
            'Too heavy for my skin, caused breakouts',
            'Loved it, my pigmentation reduced',
            'Worst product, waste of money',
            'Best moisturizer for dry skin'
        ])
    })
reviews_df = pd.DataFrame(reviews)

# Part E: Product Catalog (from Cosmetics dataset logic)
products = pd.DataFrame({
    'ProductID': range(100, 500),
    'Product_Name': [f'Product_{i}' for i in range(100,500)],
    'Suitable_Skin_Type': np.random.choice(['Oily','Dry','Combination','All'], 400),
    'Ingredients': np.random.choice(['Salicylic Acid, Niacinamide', 'Hyaluronic Acid, Ceramide', 'Vitamin C, SPF', 'Retinol, Centella'], 400),
    'Price': np.random.randint(300, 2000, 400)
})

print(f"Customers Shape: {customers.shape}")
print(f"Reviews Shape: {reviews_df.shape}")
print(f"Products Shape: {products.shape}")
print(f"Churn Rate: {customers['Churn'].mean()*100:.2f}%")

# Save files
os.makedirs('Week2_Data', exist_ok=True)
customers.to_csv('Week2_Data/customers.csv', index=False)
reviews_df.to_csv('Week2_Data/reviews.csv', index=False)
products.to_csv('Week2_Data/products.csv', index=False)

print("\nFiles Saved in Week2_Data/:")
print("- customers.csv")
print("- reviews.csv")
print("- products.csv")
print("\nHypothesis Test H1:")
h1_data = customers[(customers['Recency'] > 45)]
print(f"Customers with Recency>45: {len(h1_data)}, Churn %: {h1_data['Churn'].mean()*100:.1f}% (Expected >70% for high risk)")

print("\nWeek 2 Data Sourcing Completed!")
