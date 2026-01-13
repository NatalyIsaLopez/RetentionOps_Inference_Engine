# RetentionOps: End-to-End Churn Intelligence

### [Live Demo Link](https://share.streamlit.io/) | [Project Notebook](./exploration.ipynb)

RetentionOps is a product-oriented machine learning system designed to identify at-risk customers and provide actionable retention strategies. It moves beyond simple prediction by providing local feature explainability and a "What-If" simulation environment for Customer Success teams.



## Business Problem
Acquiring new customers is **5x more expensive** than retaining existing ones. This project provides a technical solution to identify high-risk customers in the Telecommunications sector, allowing companies to decrease churn rates through targeted, data-driven interventions.

## Data Source
The model is trained on the **IBM Telco Customer Churn Dataset**, a benchmark dataset representing a sample of 7,043 California-based customers.
* **Source:** [Kaggle / IBM Sample Data Sets](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Dataset Characteristics:** Includes 19 features covering demographics, account tenure, contract types, and service usage metrics.



## Key Features
* **Predictive Inference:** Utilizes a high-performance XGBoost model calibrated for churn probability.
* **Local Explainability:** A custom-built visualization engine showing exactly which factors (Contract, Tenure, etc.) are driving risk for an individual customer.
* **Bespoke UI/UX:** A high-contrast, dark-mode dashboard built with Streamlit and custom CSS for an operational "SaaS" feel.
* **Strategy Engine:** Context-aware recommendations (Discounts, Upsells) based on the calculated risk score.

## Tech Stack
* **Modeling:** Python 3.12, XGBoost, Scikit-Learn
* **Explainability:** Local Feature Attribution Mapping
* **Frontend:** Streamlit (Custom CSS/Bespoke Grid Layout)
* **Inference:** Model persistence via Pickle with pinned dependencies for environment stability.



## Analysis Insights
* **The "Contract" Trap:** Month-to-month contracts are the #1 leading indicator of churn, showing a 3x higher risk than long-term agreements.
* **Fiber Optic Sensitivity:** Customers on Fiber Optic tiers show higher churn variance, indicating high sensitivity to monthly pricing or service reliability.
* **Retention Sweet Spot:** Churn risk significantly drops after the 18-month tenure mark, suggesting that "Initial Loyalty" programs should focus on the first year.

## Local Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/RetentionOps-Inference-Engine.git](https://github.com/YOUR_USERNAME/RetentionOps-Inference-Engine.git)
   cd RetentionOps-Inference-Engine
