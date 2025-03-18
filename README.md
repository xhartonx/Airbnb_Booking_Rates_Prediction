# Airbnb Booking Rate Prediction Project

The goal of this project was to develop a predictive model to identify factors influencing high booking rates on Airbnb properties and provide actionable insights for Airbnb hosts and stakeholders.

---

## Table of Contents

- [Team Members](#team-members)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Exploratory Data Analysis](#Exploatory-Data-Analysis)
- [Models and Results](#models-and-results)
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

## Exploratory Data Analysis
1. **Amenities Clustering**: Listings with Cluster 2 amenities are most popular.
<img width="700" alt="cluster" src="https://github.com/user-attachments/assets/858211a4-37a4-4310-aa0a-23041981ac67">

2. **Proximity Analysis**: Listings farther from airports tend to have lower booking rates.
<img width="650" alt="violin" src="https://github.com/user-attachments/assets/16ee8f93-1ca6-47d4-8f07-aa8a6029c5e5">

3. **Price Sensitivity**: Higher prices correlate with lower booking rates.
<img width="650" alt="scatter 90" src="https://github.com/user-attachments/assets/105405d9-3d1e-4e38-83b8-c281139cf32a">

4. **Host Features**: Superhost status and responsiveness significantly influence booking rates.
<img width="600" alt="scatter plot" src="https://github.com/user-attachments/assets/4f518c9e-1e43-4ee3-ba0f-a9004f0261cc">

5. **Positive Scores**: The lack of correlation between prices and high positive scores suggests that customers are not necessarily attributing value or quality to higher-priced listings when assigning positive ratings.
<img width="600" alt="positive score" src="https://github.com/user-attachments/assets/f26bf99d-f888-4470-950c-1d9660cbf895">

---

## Models and Results

### Training and Generalization Performance
- Our team created 6 models, including Logistic, Ridge, Lasso, XGBoost, Boosting, and Random Forest. Discussing splitting the data in detail, we separated into training, testing, and validation data. Then we took the training data to train our model and predict our model on the validation data to get the AUC value. After running all the models, we plotted the models’ AUC curves on the same plot and chose the top 3 winning models with the best AUC then trained the whole data to receive the models’ probability. Finally, we averaged these models’ probability for the final prediction.

| Model              | AUC   | Description                                           |
|--------------------|-------|-------------------------------------------------------|
| Logistic Regression| 0.84  | Baseline model for comparison.                       |
| Random Forest      | 0.88  | Top-performing model for feature importance analysis. |
| Ridge Regression   | 0.86  | Penalized regression to prevent overfitting.         |
| Lasso Regression   | 0.85  | Feature selection for dimensionality reduction.      |
| XGBoost            | 0.88  | Gradient boosting for accurate predictions.          |
| Boosting           | 0.88  | Improved performance with boosting techniques.       |

- AUC for 6 models
<img width="534" alt="AUC" src="https://github.com/user-attachments/assets/aa42ae5b-735a-4616-bb7f-08ff11625849">


- The plot below uses the random forest to generate a features importance chart and see which variables are more important to the model in two aspects, giving us some insight.
<img width="530" alt="feature importance" src="https://github.com/user-attachments/assets/bc1af31b-d048-4dbe-9b11-1bd569e8dcd2">

- Ridge & Lasso model

1. Ridge model
<img width="502" alt="Ridge" src="https://github.com/user-attachments/assets/bc9d2280-e3f0-4183-baf5-5b291b20fa47">

2. Lasso model
<img width="502" alt="Lasso" src="https://github.com/user-attachments/assets/e5195cc7-1e7d-4b41-b151-1300f773eba0">

- ROC Curve

1. Logistic Regression Model
<img width="449" alt="Logistic ROC" src="https://github.com/user-attachments/assets/8c37b7a9-72f6-4f19-845a-d56edc1fe448">

2. Random Forest Model
<img width="447" alt="random forest ROC" src="https://github.com/user-attachments/assets/75859c68-e978-428c-b1c2-2fdcafbf991c">

3. Ridge Model
<img width="447" alt="Ridge ROC" src="https://github.com/user-attachments/assets/42d4927a-58ab-4cb1-a283-6c5ddd974411">

4. Lasso Model
<img width="447" alt="Lasso ROC" src="https://github.com/user-attachments/assets/53a048d0-cea2-44ee-9a8f-1964bbca9b08">

5. XGboost Model
<img width="447" alt="XGboost ROC" src="https://github.com/user-attachments/assets/060b9d0f-0eb2-4501-a697-96cae4c88e7f">

6. Boosting Model
<img width="447" alt="Boosting ROC" src="https://github.com/user-attachments/assets/21e04868-f4dd-4cdc-a157-bfc452164442">

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
