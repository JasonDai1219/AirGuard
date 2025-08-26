# AirGuard: A Clustering-Based Anomaly Detector with Real-Time User Feedback and Platform Risk Summaries

This project builds an interpretable anomaly detection system for Airbnb listings using unsupervised clustering techniques. In addition to traditional classification, we integrate a large language model (LLM) to generate human-readable explanations for each detection result, improving system transparency and user trust.

## EDA 
- **Table Creation**:
  ```sql
  CREATE OR REPLACE TABLE listings_nyc (
    id INTEGER,
    listing_url VARCHAR,
    scrape_id VARCHAR,
    last_scraped VARCHAR,
    source VARCHAR,
    name VARCHAR,
    description VARCHAR,
    neighborhood_overview VARCHAR,
    picture_url VARCHAR,
    host_id VARCHAR,
    host_url VARCHAR,
    host_name VARCHAR,
    host_since VARCHAR,
    host_location VARCHAR,
    host_about VARCHAR,
    host_response_time VARCHAR,
    host_response_rate VARCHAR,
    host_acceptance_rate VARCHAR,
    host_is_superhost VARCHAR,
    host_thumbnail_url VARCHAR,
    host_picture_url VARCHAR,
    host_neighbourhood VARCHAR,
    host_listings_count FLOAT,
    host_total_listings_count FLOAT,
    host_verifications VARCHAR,
    host_has_profile_pic VARCHAR,
    host_identity_verified VARCHAR,
    neighbourhood VARCHAR,
    neighbourhood_cleansed VARCHAR,
    neighbourhood_group_cleansed VARCHAR,
    latitude FLOAT,
    longitude FLOAT,               
    property_type VARCHAR,
    room_type VARCHAR,
    accommodates INTEGER,
    bathrooms FLOAT,
    bathrooms_text VARCHAR,
    bedrooms FLOAT,
    beds FLOAT,
    amenities VARCHAR,
    price VARCHAR,
    minimum_nights INTEGER,
    maximum_nights INTEGER,
    minimum_minimum_nights INTEGER,
    maximum_minimum_nights INTEGER,
    minimum_maximum_nights INTEGER,
    maximum_maximum_nights INTEGER,
    minimum_nights_avg_ntm FLOAT,
    maximum_nights_avg_ntm FLOAT,
    calendar_updated VARCHAR,
    has_availability VARCHAR,
    availability_30 INTEGER,
    availability_60 INTEGER,
    availability_90 INTEGER,
    availability_365 INTEGER,
    calendar_last_scraped VARCHAR,
    number_of_reviews INTEGER,
    number_of_reviews_ltm INTEGER,
    number_of_reviews_l30d INTEGER,
    availability_eoy INTEGER,
    number_of_reviews_ly INTEGER,
    estimated_occupancy_l365d INTEGER,
    first_review VARCHAR,
    last_review VARCHAR,
    review_scores_rating FLOAT,
    review_scores_accuracy FLOAT,
    review_scores_cleanliness FLOAT,
    review_scores_checkin FLOAT,
    review_scores_communication FLOAT,
    review_scores_location FLOAT,
    review_scores_value FLOAT,
    license VARCHAR,
    calculated_host_listings_count VARCHAR,
    calculated_host_listings_count_entire_homes VARCHAR,
    calculated_host_listings_count_private_rooms VARCHAR,
    calculated_host_listings_count_shared_rooms VARCHAR,
    reviews_per_month VARCHAR,
    instant_bookable VARCHAR);
  
- **Data Cleaning**:
  Data Cleaning Logic  
  The SQL query below performs the following steps:  
  
  1. **Field Selection**  
     - Chooses relevant columns such as `room_type`, `property_type`, `accommodates`, `bathrooms`, `bedrooms`, `beds`, etc.  
     - Focuses on fields that are most useful for analysis.  
  
  2. **Data Cleaning**  
     - **Price (`price_cleaned`)**: Removes `$` and `,`, then converts to `FLOAT`.  
     - **Host Acceptance Rate (`host_acceptance_rate_cleaned`)**: Strips `%` and converts to numeric.  
     - **Reviews per Month (`reviews_per_month_cleaned`)**: Casts directly to numeric.  
  
  3. **Filtering Rules**  
     - Excludes rows where `room_type` or `price` is missing.  
     - Removes records where `host_acceptance_rate` is empty or set to `'N/A'`.  
     - Keeps only listings with positive prices (`> 0`).  
  
  4. **Row Indexing**  
     - Adds a unique sequential row number (`row_index`) using  
       `ROW_NUMBER() OVER (ORDER BY price_cleaned)` for easier sorting and referencing.  
  
  After these transformations, the result is a **cleaned dataset** ready for Exploratory Data Analysis (EDA) and visualization.  

  ```sql
  WITH t1 AS (
  SELECT
    room_type,
    property_type,
    accommodates,
    bathrooms,
    bedrooms,
    beds,
    TRY_CAST(REPLACE(REPLACE(price, '$', ''), ',', '') AS FLOAT) AS price_cleaned,
    minimum_nights,
    maximum_nights,
    availability_365,
    host_is_superhost,
    host_listings_count,
    instant_bookable,
    TRY_CAST(REPLACE(host_acceptance_rate, '%', '') AS FLOAT) AS host_acceptance_rate_cleaned,
    latitude,
    longitude,
    TRY_CAST(reviews_per_month AS FLOAT) AS reviews_per_month_cleaned
  FROM listings_nyc
  WHERE 
    room_type IS NOT NULL
    AND price IS NOT NULL
    AND host_acceptance_rate IS NOT NULL
    AND host_acceptance_rate NOT IN ('', 'N/A')
    AND TRY_CAST(REPLACE(REPLACE(price, '$', ''), ',', '') AS FLOAT) > 0)
  SELECT 
    ROW_NUMBER() OVER (ORDER BY price_cleaned) AS row_index, t1.*
  FROM t1;

- **Data Visualization**:

## Key Features

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

## System Workflow

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

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- openai (for LLM explanation)
- seaborn / matplotlib (optional for visualization)

## Usage

```python
from model import classify_listing_from_raw_input
from explain import generate_natural_language_explanation

# Step 1: Classify
result = classify_listing_from_raw_input(user_input, ...)

# Step 2: If anomaly, generate explanation
if result["type"] == "anomaly":
    explanation = generate_natural_language_explanation(user_input, result)
    print(explanation)
