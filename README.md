# AirGuard: A Clustering-Based Anomaly Detector with Real-Time User Feedback and Platform Risk Summaries

This project builds an interpretable anomaly detection system for Airbnb listings using unsupervised clustering techniques. In addition to traditional classification, we integrate a large language model (LLM) to generate human-readable explanations for each detection result, improving system transparency and user trust.

## EDA 

## 📌 Key Features

- **Unsupervised Anomaly Detection**: 
  - Uses DBSCAN for clustering
  - Performs feature selection via RandomForest based on initial cluster labels
  - Classifies listings as `typical` or `anomaly` based on distance to cluster center and cluster rarity

- **Robust Evaluation**:
  - Synthesized evaluation set of 100+ listings (typical + anomaly)
  - F1 score on anomalies > **0.85**, with **recall = 1.00**
  - Typical listings F1 score also > **0.83**

- **LLM-based Explanation Module**:
  - For each listing, a natural language explanation is generated based on:
    - Key feature values
    - Cluster distance
    - Deviations from cluster norms
  - Example output:
    ```
    This listing is classified as an anomaly because it has a very high price ($980) and an unusually high number of bedrooms (8) for a small accommodation capacity (2). It also differs significantly from nearby cluster members in availability and host activity.
    ```

## 🧠 System Workflow

1. **Data Preprocessing**: Handle missing values, derive log-transformed and ratio-based features, one-hot encode categorical variables.
2. **Initial Clustering**: PCA + DBSCAN to label listings and initialize clusters.
3. **Feature Selection**: RandomForest classifier on cluster labels to extract top informative features.
4. **Re-clustering**: Re-run DBSCAN on selected features and standardize.
5. **Classification**: For each new listing, determine:
    - Closest cluster
    - Distance to cluster center
    - Whether it falls outside 95% confidence radius or in a rare cluster
6. **Natural Language Explanation (Optional)**:
    - Listing’s features + cluster deviation are passed to an LLM (e.g., GPT-4) to generate a detailed explanation.

## 🛠️ Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- openai (for LLM explanation)
- seaborn / matplotlib (optional for visualization)

## 🚀 Usage

```python
from model import classify_listing_from_raw_input
from explain import generate_natural_language_explanation

# Step 1: Classify
result = classify_listing_from_raw_input(user_input, ...)

# Step 2: If anomaly, generate explanation
if result["type"] == "anomaly":
    explanation = generate_natural_language_explanation(user_input, result)
    print(explanation)
