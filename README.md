# Airbnb Booking Rate Prediction Project

This repository contains the final project for **BUDT 758T**, completed in Spring 2024. The goal of this project was to develop a predictive model to identify factors influencing high booking rates on Airbnb properties and provide actionable insights for Airbnb hosts and stakeholders.

---

## Table of Contents

- [Team Members](#team-members)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models and Results](#models-and-results)
- [Insights](#insights)
- [Reflection](#reflection)
- [How to Run](#how-to-run)
- [Contributions](#contributions)
- [License](#license)

---

## Team Members

- **Yiting Lai**: Team Lead, Data Cleaning, Feature Engineering, Modeling, EDA, Text Mining, R Code Organization
- **Ching Yu Ting**: Data Cleaning, Feature Engineering, Modeling
- **Cheng-Feng Lin**: Data Cleaning, Feature Engineering, Modeling
- **Kelan Quan**: Feature Engineering, Text Mining
- **I-Chia Yeh**: Data Cleaning, Feature Engineering, Text Mining, EDA

---

## Project Overview

The hospitality industry is highly competitive, and optimizing Airbnb listings is crucial for hosts to maximize bookings. This project aimed to predict high booking rates by analyzing various factors, including property attributes, host characteristics, and guest feedback.

### Target Audience
- Airbnb Hosts
- Property Management Companies
- Real Estate Investors
- Airbnb Platform Stakeholders

### Value Proposition
- **Granular Insights**: Understand key factors affecting booking success.
- **Optimized Pricing**: Tailored strategies for competitive pricing.
- **Guest Experience Enhancement**: Improve amenities and services based on data-driven insights.
- **Location Selection**: Identify profitable markets for expansion.

---

## Dataset

The dataset includes features such as property characteristics, host details, and guest feedback. Key variables include:
- **Property Attributes**: Number of bedrooms, amenities, etc.
- **Host Information**: Superhost status, response rate, etc.
- **Guest Feedback**: Text mining on property descriptions, reviews, and more.

---

## Methodology

1. **Data Cleaning and Preparation**
   - Handled missing values and created dummy variables for categorical features.
   - Conducted feature engineering using techniques like clustering and text mining.

2. **Modeling**
   - Evaluated six models: Logistic Regression, Ridge, Lasso, Random Forest, XGBoost, and Boosting.
   - Used cross-validation (10-fold) for model selection.
   - Selected top three models based on AUC (Area Under the Curve).

3. **Evaluation**
   - Final models: Random Forest, XGBoost, Boosting.
   - AUC for the best model: **0.88**.

4. **Insights**
   - Text mining highlighted impactful keywords for optimizing property descriptions.
   - Proximity to airports and popular attractions significantly influences booking rates.

---

## Models and Results

| Model              | AUC   | Description                                           |
|--------------------|-------|-------------------------------------------------------|
| Logistic Regression| 0.84  | Baseline model for comparison.                       |
| Random Forest      | 0.88  | Top-performing model for feature importance analysis. |
| Ridge Regression   | 0.86  | Penalized regression to prevent overfitting.         |
| Lasso Regression   | 0.85  | Feature selection for dimensionality reduction.      |
| XGBoost            | 0.88  | Gradient boosting for accurate predictions.          |
| Boosting           | 0.88  | Improved performance with boosting techniques.       |

---

## Insights

1. **Amenities Clustering**: Listings with Cluster 2 amenities are most popular.
2. **Proximity Analysis**: Listings farther from airports tend to have lower booking rates.
3. **Price Sensitivity**: Higher prices correlate with lower booking rates.
4. **Host Features**: Superhost status and responsiveness significantly influence booking rates.

---

## Reflection

### Strengths
- Effective collaboration with clear communication.
- Comprehensive modeling and feature engineering.

### Challenges
- Handling messy and high-dimensional data.
- Balancing model complexity with interpretability.

### Future Work
- Prioritize text mining earlier in the process.
- Assign weights to features for improved modeling.

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/airbnb-booking-prediction.git
