USE DATABASE LISTINGS;

CREATE OR REPLACE FILE FORMAT my_csv_format
TYPE = 'CSV'
FIELD_OPTIONALLY_ENCLOSED_BY = '"'
SKIP_HEADER = 1
ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE
NULL_IF = ('NULL', 'null');


CREATE OR REPLACE TABLE listings_nyc (
    id INTEGER,
    listing_url STRING,
    scrape_id STRING,
    last_scraped STRING,
    source STRING,
    name STRING,
    description STRING,
    neighborhood_overview STRING,
    picture_url TEXT(255),
    host_id STRING,
    host_url TEXT(255),
    host_name STRING,
    host_since STRING,
    host_location STRING,
    host_about STRING,
    host_response_time STRING,
    host_response_rate STRING,
    host_acceptance_rate STRING,
    host_is_superhost STRING,
    host_thumbnail_url TEXT(255),
    host_picture_url TEXT(255),
    host_neighbourhood STRING,
    host_listings_count FLOAT,
    host_total_listings_count FLOAT,
    host_verifications STRING,
    host_has_profile_pic STRING,
    host_identity_verified STRING,
    neighbourhood STRING,
    neighbourhood_cleansed STRING,
    neighbourhood_group_cleansed STRING,
    latitude FLOAT,
    longtitude FLOAT,
    property_type STRING,
    room_type STRING,
    accommodates INTEGER,
    bathrooms FLOAT,
    bathrooms_text STRING,
    bedrooms FLOAT,
    beds FLOAT,
    amenities STRING,
    price STRING,
    minimum_nights INTEGER,
    maximum_nights INTEGER,
    minimum_minimum_nights INTEGER,
    maximum_minimum_nights INTEGER,
    minimum_maximum_nights INTEGER,
    maximum_maximum_nights INTEGER,
    minimum_nights_avg_ntm FLOAT,
    maximum_nights_avg_ntm FLOAT,
    calendar_updated STRING,
    has_availability STRING,
    availability_30 INTEGER,
    availability_60 INTEGER,
    availability_90 INTEGER,
    availability_365 INTEGER,
    calendar_last_scraped STRING,
    number_of_reviews INTEGER,
    number_of_reviews_ltm INTEGER,
    number_of_reviews_l30d INTEGER,
    availability_eoy INTEGER,
    number_of_reviews_ly INTEGER,
    estimated_occupancy_l365d INTEGER,
    first_review STRING,
    last_review STRING,
    review_scores_rating FLOAT,
    review_scores_accuracy FLOAT,
    review_scores_cleanliness FLOAT,
    review_scores_checkin FLOAT,
    review_scores_communication FLOAT,
    review_scores_location FLOAT,
    review_scores_value FLOAT,
    license STRING,
    calculated_host_listings_count STRING,
    calculated_host_listings_count_entire_homes STRING,
    calculated_host_listings_count_private_rooms STRING,
    calculated_host_listings_count_shared_rooms STRING,
    reviews_per_month STRING,
    instant_bookable STRING
);


LIST @my_s3_stage;

COPY INTO listings_nyc
FROM @my_s3_stage/listings.csv
ON_ERROR = 'CONTINUE'
FILE_FORMAT = my_csv_format;


SELECT * FROM listings_nyc LIMIT 10;

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
  longtitude,



  TRY_CAST(reviews_per_month AS FLOAT) AS reviews_per_month_cleaned

FROM listings_nyc
WHERE 
  room_type IS NOT NULL
  AND price IS NOT NULL
  AND host_acceptance_rate IS NOT NULL
  AND host_acceptance_rate NOT IN ('', 'N/A') 
  AND TRY_CAST(REPLACE(REPLACE(price, '$', ''), ',', '') AS FLOAT) > 0;