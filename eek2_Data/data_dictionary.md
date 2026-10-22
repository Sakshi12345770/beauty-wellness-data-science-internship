# Data Dictionary - Week 2

## customers.csv
- CustomerID: Unique ID
- Age: 18-55
- Recency: Days since last purchase (Key for Churn)
- Frequency: Total orders
- Monetary: Total spend
- Skin_Type: Oily/Dry/Combination/Sensitive (15% missing - will impute)
- Skin_Concern: Acne/Pigmentation etc.
- Churn: 1=Churned (Recency>45), 0=Active
- Wishlist_Count: Behavioural feature

## reviews.csv
- CustomerID, ProductID, Rating (1-5), Review_Text (for Sentiment Analysis)

## products.csv
- ProductID, Product_Name, Suitable_Skin_Type, Ingredients (for Content-based Filtering)

## Data Sources Used
1. Sephora Products & Reviews - https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews
2. Customer Personality Analysis - https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
3. Cosmetics Datasets - https://www.kaggle.com/datasets/kingabzpro/cosmetics-datasets

## Quality Assessment
- Completeness: 85% (Skin Type missing 15%)
- Accuracy: Valid ranges checked
- Relevance: 9/10 for churn + recommendation
