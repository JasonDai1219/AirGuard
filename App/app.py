import streamlit as st
# 🔑 必须是脚本里第一个 Streamlit 调用，且只能调用一次
st.set_page_config(page_title="Airbnb Listing Classifier", page_icon="🏠", layout="centered")

import json
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# ============== 加载工件 ==============
@st.cache_resource
def load_artifacts():
    try:
        scaler = joblib.load("scaler.pkl")
        with open("top_features.json","r") as f:
            top_features = json.load(f)
        X_scaled = np.load("X_scaled.npy")
        cluster_labels = np.load("cluster_labels.npy")
    except Exception as e:
        raise RuntimeError(f"加载工件失败：{e}")
    return scaler, top_features, X_scaled, cluster_labels

# ============== 判定函数（你的原函数） ==============
def classify_listing_from_raw_input(user_input_raw, top_features, scaler, X_scaled, cluster_labels, rare_threshold=0.04):
    df_input = pd.DataFrame([user_input_raw])

    # 强制数值类型/默认值
    df_input["PRICE"] = df_input["PRICE"].astype(float)
    df_input["REVIEWS_PER_MONTH"] = df_input["REVIEWS_PER_MONTH"].astype(float)
    df_input["HOST_LISTINGS_COUNT"] = df_input["HOST_LISTINGS_COUNT"].astype(float)
    df_input["AVAILABILITY_365"] = df_input["AVAILABILITY_365"].astype(float)
    df_input["BEDROOMS"] = df_input["BEDROOMS"].fillna(1)
    df_input["BEDS"] = df_input["BEDS"].fillna(1)

    # 衍生特征
    df_input["LOG_PRICE"] = np.log1p(df_input["PRICE"])
    df_input["LOG_REVIEWS_PER_MONTH"] = np.log1p(df_input["REVIEWS_PER_MONTH"])
    df_input["LOG_HOST_LISTINGS_COUNT"] = np.log1p(df_input["HOST_LISTINGS_COUNT"])
    df_input["LOG_AVAILABILITY_365"] = np.log1p(df_input["AVAILABILITY_365"])
    df_input["LISTING_DENSITY"] = df_input["REVIEWS_PER_MONTH"] / (df_input["AVAILABILITY_365"] + 1)
    df_input["BEDROOM_BED_RATIO"] = df_input["BEDROOMS"] / (df_input["BEDS"] + 1)

    # One-hot
    room_dummies = pd.get_dummies(df_input.get("ROOM_TYPE", pd.Series(dtype=str)), prefix="ROOM_TYPE")
    prop_dummies = pd.get_dummies(df_input.get("PROPERTY_TYPE", pd.Series(dtype=str)), prefix="PROPERTY_TYPE")
    df_input = pd.concat([df_input, room_dummies, prop_dummies], axis=1)

    # 对齐特征
    for col in top_features:
        if col not in df_input.columns:
            df_input[col] = 0

    X_user = df_input[top_features]
    X_user_scaled = scaler.transform(X_user)

    # 距离与簇中心
    from numpy import ndarray
    cluster_sizes = Counter(cluster_labels)
    cluster_centers, cluster_distances = {}, {}

    unique_labels = set(cluster_labels.tolist() if isinstance(cluster_labels, ndarray) else cluster_labels)
    for label in unique_labels:
        if label == -1:
            continue
        mask = (cluster_labels == label)
        cluster_points = X_scaled[mask]
        center = cluster_points.mean(axis=0)
        cluster_centers[label] = center
        dist = euclidean_distances(X_user_scaled, center.reshape(1, -1))[0][0]
        cluster_distances[label] = float(dist)

    if not cluster_distances:
        return {"type": "anomaly", "reason": "No clusters available."}

    closest_cluster = min(cluster_distances, key=cluster_distances.get)
    closest_distance = cluster_distances[closest_cluster]
    cluster_ratio = cluster_sizes[closest_cluster] / len(cluster_labels)

    own_distances = euclidean_distances(
        X_scaled[cluster_labels == closest_cluster],
        cluster_centers[closest_cluster].reshape(1, -1)
    )
    abnormal_cutoff = float(np.percentile(own_distances, 95))

    if closest_distance > abnormal_cutoff:
        label_type = "anomaly"
    elif cluster_ratio < rare_threshold:
        label_type = "rare"
    else:
        label_type = "typical"

    return {
        "type": label_type,
        "closest_cluster": int(closest_cluster),
        "cluster_size_ratio": round(float(cluster_ratio), 4),
        "distance_to_cluster_center": round(float(closest_distance), 4),
        "abnormal_cutoff": round(float(abnormal_cutoff), 4),
        "all_cluster_distances": dict(sorted(cluster_distances.items(), key=lambda kv: kv[1]))
    }

# ============== 评估函数 ==============
def evaluate_anomaly_detector(test_listings, classify_func, top_features, scaler, X_scaled, cluster_labels):
    y_true, y_pred = [], []
    for listing in test_listings:
        listing = listing.copy()
        label = listing.pop("LABEL")
        y_true.append(label)
        try:
            result = classify_func(
                user_input_raw=listing,
                top_features=top_features,
                scaler=scaler,
                X_scaled=X_scaled,
                cluster_labels=cluster_labels
            )
            y_pred.append(result.get("type", "error"))
        except Exception:
            y_pred.append("error")

    labels_order = ["typical", "rare", "anomaly"]
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    report = classification_report(y_true, y_pred, labels=labels_order, output_dict=True)
    return (
        pd.DataFrame(cm, index=labels_order, columns=labels_order),
        pd.DataFrame(report).T,
        y_true,
        y_pred,
    )

# ============== UI ==============
st.title("🏠 Airbnb Listing Classifier")

# 加载工件
try:
    scaler, top_features, X_scaled, cluster_labels = load_artifacts()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

mode = st.sidebar.radio("模式", ["单条预测", "批量评估"], index=0)

if mode == "单条预测":
    with st.form("input_form", clear_on_submit=False):
        st.subheader("基本数值特征")
        col1, col2 = st.columns(2)
        with col1:
            price = st.number_input("PRICE (USD)", min_value=0.0, value=120.0, step=1.0)
            bedrooms = st.number_input("BEDROOMS", min_value=0.0, value=1.0, step=1.0)
            beds = st.number_input("BEDS", min_value=0.0, value=1.0, step=1.0)
            reviews_per_month = st.number_input("REVIEWS_PER_MONTH", min_value=0.0, value=0.5, step=0.1)
        with col2:
            availability_365 = st.number_input("AVAILABILITY_365", min_value=0.0, value=120.0, step=1.0)
            host_listings_count = st.number_input("HOST_LISTINGS_COUNT", min_value=0.0, value=1.0, step=1.0)

        st.subheader("类别特征")
        room_type = st.text_input("ROOM_TYPE（如 Entire home/apt）", value="Entire home/apt")
        property_type = st.text_input("PROPERTY_TYPE（如 Apartment）", value="Apartment")

        rare_threshold = st.slider("Rare 阈值（簇占比）", 0.0, 0.2, 0.04, 0.01)

        submitted = st.form_submit_button("🚀 预测")

    if submitted:
        user_input_raw = {
            "PRICE": price,
            "REVIEWS_PER_MONTH": reviews_per_month,
            "HOST_LISTINGS_COUNT": host_listings_count,
            "AVAILABILITY_365": availability_365,
            "BEDROOMS": bedrooms,
            "BEDS": beds,
            "ROOM_TYPE": room_type,
            "PROPERTY_TYPE": property_type,
        }
        with st.spinner("推理中…"):
            result = classify_listing_from_raw_input(
                user_input_raw=user_input_raw,
                top_features=top_features,
                scaler=scaler,
                X_scaled=X_scaled,
                cluster_labels=cluster_labels,
                rare_threshold=rare_threshold
            )

        st.divider()
        st.subheader("结果")
        label = result.get("type", "unknown")
        if label == "typical":
            st.success(f"判定：**{label.upper()}**")
        elif label == "rare":
            st.warning(f"判定：**{label.upper()}**")
        elif label == "anomaly":
            st.error(f"判定：**{label.upper()}**")
        else:
            st.info(f"判定：**{label.upper()}**")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Closest Cluster", result.get("closest_cluster", "-"))
            st.metric("Cluster Size Ratio", result.get("cluster_size_ratio", "-"))
        with c2:
            st.metric("Distance to Center", result.get("distance_to_cluster_center", "-"))
            st.metric("Cluster 95% Cutoff", result.get("abnormal_cutoff", "-"))

        if "all_cluster_distances" in result:
            st.subheader("各簇距离（从近到远）")
            df_dist = pd.DataFrame(
                [{"cluster": k, "distance": v} for k, v in result["all_cluster_distances"].items()]
            )
            st.dataframe(df_dist, use_container_width=True)

else:  # 批量评估
    st.subheader("批量评估（confusion matrix / classification report）")

    default_path = Path("full_test_listings.json")
    test_listings = None

    if default_path.exists():
        with open(default_path, "r") as f:
            test_listings = json.load(f)
        st.caption("已自动加载：full_test_listings.json")
    else:
        up = st.file_uploader("上传 full_test_listings.json（list[dict]，每条包含 LABEL）", type=["json"])
        if up is not None:
            test_listings = json.load(up)

    if test_listings is not None:
        with st.spinner("评估中…"):
            cm_df, report_df, y_true, y_pred = evaluate_anomaly_detector(
                test_listings=test_listings,
                classify_func=classify_listing_from_raw_input,
                top_features=top_features,
                scaler=scaler,
                X_scaled=X_scaled,
                cluster_labels=cluster_labels
            )

        st.write("### 混淆矩阵")
        st.dataframe(cm_df, use_container_width=True)
        st.write("### 分类报告")
        st.dataframe(report_df, use_container_width=True)
    else:
        st.info("请上传或放置 `full_test_listings.json` 后再开始评估。")
