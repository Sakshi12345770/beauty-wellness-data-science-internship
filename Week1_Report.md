 Data Science Orientation & Problem Definition in Beauty & Wellness

**Intern:** Sakshi Suresh Wankhade  
**Internship:** Data Science with Python - Henry Harvin (Skill India - MSDE)  
**Project:** Predicting Customer Churn & Personalized Skincare Recommendation System  


### 1. Business Problem Statement
How can a beauty D2C brand predict which customers are likely to churn in the next 30-60 days and recommend personalized skincare products to increase customer retention and Customer Lifetime Value (CLV)?

In India, 60%+ beauty e-commerce customers do not make a repeat purchase. The primary reason is irrelevant product recommendations. For example, recommending a heavy oil-based moisturizer to a customer with oily, acne-prone skin leads to dissatisfaction. If we can identify customer skin type and concern using data, we can recommend the right product and improve retention.

### 2. Relevance in Beauty & Wellness Industry
- **Market Size:** India beauty and personal care market is $20B+ and growing at 25% CAGR.
- **High CAC:** Acquiring a new customer is 5x more expensive than retaining an existing one.
- **Demand for Personalization:** According to McKinsey 2024, 71% of beauty consumers expect personalized recommendations. Brands like Nykaa, Sephora use AI to increase sales by 25%.
- **Review Driven:** 85% of beauty purchases are influenced by online reviews and ratings.

### 3. Research Questions (RQ)
RQ1: What factors cause churn? (Recency - days since last purchase, Frequency, Monetary, Avg Rating)
RQ2: Can we segment customers based on skin type (Oily, Dry, Combination) for targeted marketing?
RQ3: Do customers with similar skin types give similar ratings to similar products? (To validate Collaborative Filtering)

### 4. Hypothesis
H1: Customers who are inactive for more than 45 days and have given ratings less than 3 have 70% higher probability of churn.
H2: Skin type-based personalized recommendations will have higher CTR than generic bestseller recommendations.

### 5. Data Required
1. Customer Demographic: Age, Location, Skin Type, Skin Concern (Acne, Pigmentation)
2. Transactional: CustomerID, OrderDate, ProductID, Price, Discount
3. Behavioural: Session time on app, Wishlist adds, Product views
4. Product: ProductID, Category, Ingredients, Suitable Skin Type
5. Reviews: Rating (1-5), Review Text, Date
6. Target Variable: Churn (1 = No purchase in last 60 days, 0 = Active)

### 6. Proposed Approach
**Phase 1: EDA** - RFM Analysis, Churn distribution, WordCloud for negative reviews, Correlation heatmap
**Phase 2: Preprocessing & Feature Engineering** - Handle missing skin type (Mode imputation), Text cleaning (NLTK), Create features: Recency, Sentiment Score using VADER, Days since last wishlist
**Phase 3: Modeling**
- Churn Prediction: Logistic Regression, Random Forest, XGBoost (Metrics: Accuracy, Precision, Recall, ROC-AUC)
- Recommendation System: Hybrid - Content-Based (Ingredients) + Collaborative Filtering (Surprise library)
- Sentiment Analysis: VADER / BERT for review classification
**Phase 4: Dashboard & Marketing Action** - High churn risk customers to receive personalized discount + bundle offer via dashboard

### 7. Tools & Libraries
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, NLTK, Surprise, XGBoost

### 8. Expected Business Impact
This system will help reduce customer churn by 15-20%, increase repeat purchase rate, and improve Customer Lifetime Value (CLV) through data-driven personalized marketing.

### 9. Datasets for Next Phase
- Sephora Products and Skincare Reviews (Kaggle)
- Amazon Beauty Product Ratings (Kaggle)
- Customer Personality Analysis Dataset (Kaggle)
