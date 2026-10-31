
# ✈️ Impact of Conflict and Flight Route Disruptions on Airline Ticket Prices and Flight Cancellations Using Machine Learning

## 📌 Project Overview

This project analyses the impact of geopolitical conflicts, flight route disruptions, and changes in aviation fuel prices on airline ticket prices and flight cancellations using Data Science and Machine Learning.

The project uses Python-based data analysis, Exploratory Data Analysis (EDA), Linear Regression, and classification algorithms to understand ticket price patterns and predict flight cancellation status.

## 🎯 Project Objectives

- Analyse airline ticket price variations across different conflict phases.
- Study the relationship between jet fuel prices and fuel surcharges.
- Understand the impact of route distance and fuel costs.
- Perform data cleaning and preprocessing.
- Predict total airline ticket prices using Linear Regression.
- Predict flight cancellations using classification algorithms.
- Compare machine learning models using evaluation metrics.

## 📊 Dataset Information

The project dataset contains:

| Dataset Property | Details |
|---|---|
| Total Records | 3,132 |
| Total Columns | 23 |
| Time Period | January 2019 – March 2026 |
| Distinct Monthly Periods | 87 |
| Missing Values | 0 |
| Duplicate Records | 0 |
| Cancelled Records | 68 |
| Rerouted Records | 116 |

### Important Features

- Airline
- Origin City
- Destination City
- Aircraft Type
- Conflict Phase
- Original Distance
- Actual Distance
- Jet Fuel Price
- Fuel Consumption
- Fuel Surcharge
- Total Ticket Price
- Flight Cancellation Status

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed:

1. Distribution of Total Ticket Prices
2. Average Ticket Price by Conflict Phase
3. Jet Fuel Price vs Fuel Surcharge
4. Route Distance and Additional Fuel Cost
5. Monthly Ticket Price Trends
6. Correlation Analysis of Numerical Variables

## 🤖 Machine Learning Algorithms

### Regression Model
- Linear Regression – Used to predict total airline ticket prices.

### Classification Models
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Random Forest Classifier

## 📈 Model Performance

### Linear Regression Results

| Evaluation Metric | Result |
|---|---:|
| MAE | 69.41 USD |
| MSE | 14,081.44 |
| RMSE | 118.67 USD |
| R² Score | 0.9168 |

### Classification Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 93.14% | 20.41% | 71.43% | 0.3175 | 0.9497 |
| Decision Tree | 92.82% | 21.82% | 85.71% | 0.3478 | 0.9081 |
| KNN | 97.61% | 40.00% | 14.29% | 0.2105 | 0.8326 |
| Random Forest | 94.74% | 25.64% | 71.43% | 0.3774 | 0.9691 |

**Best Classification Model:** Random Forest achieved the highest F1-score of 0.3774 and ROC-AUC of 0.9691 among the four tested classifiers.

## 📉 Key Findings

- Average ticket price in the dataset is approximately 642.65 USD.
- Ticket prices vary across different conflict phases.
- Jet fuel prices show a positive relationship with fuel surcharges.
- Linear Regression achieved an R² score of 0.9168.
- Random Forest performed best based on F1-score among the classification models.
- Flight cancellations represent approximately 2.17% of the dataset, indicating class imbalance.

## ⚙️ Project Workflow

1. Dataset Collection
2. Data Loading
3. Data Cleaning
4. Exploratory Data Analysis
5. Data Preprocessing
6. Feature Engineering
7. Model Training
8. Model Evaluation
9. Results and Visualization

## 📁 Project Structure

```text
Airline-Route-Cost-Impact-Analysis/
│
├── README.md
├── airline_route_analysis.py
├── route_cost_impact.csv
├── final report.pdf
│
└── graphs/
    ├── ticket_price_distribution.png
    ├── conflict_phase_analysis.png
    ├── fuel_price_surcharge.png
    ├── regression_prediction.png
    ├── logistic_confusion_matrix.png
    ├── decision_tree_confusion_matrix.png
    ├── knn_confusion_matrix.png
    ├── random_forest_confusion_matrix.png
    └── classification_comparison.png
```

## 🏁 Conclusion

This project demonstrates how Data Science and Machine Learning techniques can be applied to analyse airline ticket prices and flight cancellation patterns.

The analysis identifies relationships between fuel prices, route disruptions, conflict phases, and ticket prices. The regression and classification models provide useful predictive results on the supplied dataset.

The project also highlights the importance of selecting appropriate evaluation metrics when dealing with imbalanced datasets.

## ⚠️ Limitations

- Dataset provenance has not been independently verified.
- Flight cancellation records are relatively rare.
- Model results are based on a single train-test split.
- Results should not be treated as universal conclusions about the airline industry.

## 👨‍💻 Author

**Manikantasai**

B.Tech – Computer Science and Engineering

Lovely Professional University

**Project Domain:** Data Science & Machine Learning
