import streamlit as st
# 🔑 set_page_config 必须最先且只调用一次
st.set_page_config(page_title="AirGuard – Airbnb Listing Classifier", page_icon="🏠", layout="wide")

# ================== 轻量样式（卡片/发光/顶部栏/字体） ==================
st.markdown("""
<style>
:root { --glass: rgba(255,255,255,0.06); --glass2: rgba(255,255,255,0.03); }
.block-container { padding-top: 0.6rem; }
.topbar {position: sticky; top: 0; z-index: 100; backdrop-filter: blur(8px);
  background: rgba(11,12,16,0.65); border-bottom: 1px solid rgba(255,255,255,0.06); padding: 10px 8px;}
.topbar .brand {font-weight:700; letter-spacing:.5px;}
.kcard {border: 1px solid var(--glass); background: linear-gradient(180deg, var(--glass2), rgba(255,255,255,0.02));
  border-radius: 16px; padding: 16px; }
.glow {box-shadow: 0 0 0px rgba(124,92,255,0.0);}
.glow:hover {box-shadow: 0 0 22px rgba(124,92,255,0.25); transition: box-shadow .25s ease;}
.section-title {font-size: 20px; font-weight: 700; margin: 6px 0 10px;}
.small {opacity:.8; font-size: 13px;}
.footer {opacity:.65; font-size:12px; padding:24px 0 8px 0; text-align:center;}
.badge {display:inline-block; padding:4px 10px; border-radius:999px; font-weight:700; font-size:13px;}
.badge-green {background:#17351f; color:#8ef0a6; border:1px solid #214f30;}
.badge-yellow {background:#3c3310; color:#ffe27a; border:1px solid #5a4a16;}
.badge-red {background:#3a1518; color:#ff9aa2; border:1px solid #5a1f24;}
</style>
<div class="topbar">
  <span class="brand">🏠 AirGuard</span>
  <span style="opacity:.6; margin-left:8px;">Anomaly & Risk Insight</span>
</div>
<div style="padding: 10px 0 6px 0;">
  <div style="font-size:28px; font-weight:800;">城市级房源异常检测</div>
  <div class="small">按城市加载聚类基线 · 口语化解释 · 一键评估</div>
</div>
""", unsafe_allow_html=True)

# ================== Imports ==================
import json
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import hashlib

# ================== 城市配置 ==================
CITY_OPTIONS = {
    "NYC": {"label": "纽约 NYC", "default_rare_threshold": 0.04},
    "SF":  {"label": "旧金山 SF", "default_rare_threshold": 0.05},
    "LA":  {"label": "洛杉矶 LA", "default_rare_threshold": 0.05},
    "SEA": {"label": "西雅图 SEA", "default_rare_threshold": 0.05},
}

# ============== 工件加载 ==============
@st.cache_resource
def load_artifacts(city: str):
    base = Path(__file__).parent / city
    try:
        scaler = joblib.load(base / "scaler.pkl")
        with open(base / "top_features.json","r") as f:
            top_features = json.load(f)
        X_scaled = np.load(base / "X_scaled.npy")
        cluster_labels = np.load(base / "cluster_labels.npy")
    except Exception as e:
        raise RuntimeError(f"[{city}] 加载工件失败：{e}")
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

    # 在返回结果前，先算 p_in_cluster
    p_in_cluster = percentile_rank(
        euclidean_distances(X_scaled[cluster_labels == closest_cluster],
                            cluster_centers[closest_cluster].reshape(1, -1)).reshape(-1),
        closest_distance
    )

    if closest_distance > abnormal_cutoff:
        label_type = "anomaly"
    elif p_in_cluster >= 80:   # 👈 新增逻辑
        label_type = "rare"
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
        "all_cluster_distances": dict(sorted(cluster_distances.items(), key=lambda kv: kv[1])),
        "percentile_in_cluster": round(float(p_in_cluster), 2)

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

# ========= 距离索引 & 百分位工具 =========
@st.cache_resource
def build_distance_index(X_scaled: np.ndarray, cluster_labels: np.ndarray):
    labels = np.unique(cluster_labels)
    labels = labels[labels != -1]
    centers = {}
    dists_by_label = {}
    for lb in labels:
        pts = X_scaled[cluster_labels == lb]
        center = pts.mean(axis=0)
        centers[lb] = center
        d = euclidean_distances(pts, center.reshape(1, -1)).reshape(-1)
        dists_by_label[lb] = d
    global_dists = np.concatenate(list(dists_by_label.values())) if dists_by_label else np.array([])
    return centers, dists_by_label, global_dists

def percentile_rank(arr: np.ndarray, value: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float((arr <= value).mean() * 100.0)

def size_band_text(ratio: float) -> str:
    if ratio >= 0.20:   # >=20%
        return "常见"
    if ratio >= 0.05:   # 5%~20%
        return "不常见"
    return "很少见"

def humanize_diffs(top_diffs):
    name_map = {
        "LOG_PRICE": "价格（对数）", "PRICE": "价格",
        "BEDROOM_BED_RATIO": "卧室/床位比例",
        "LISTING_DENSITY": "活跃度（评论/可订天数）",
        "LOG_HOST_LISTINGS_COUNT": "房东上架数（对数）",
        "LOG_AVAILABILITY_365": "可订天数（对数）",
        "LOG_REVIEWS_PER_MONTH": "月均评论（对数）",
    }
    friendly = []
    for r in top_diffs:
        z = r["z_diff"]
        if abs(z) >= 2.5:
            level = "明显"
        elif abs(z) >= 1.5:
            level = "有点"
        else:
            level = "轻微"
        friendly.append({
            "feature": r["feature"],
            "name": name_map.get(r["feature"], r["feature"]),
            "direction": "偏高" if z > 0 else "偏低",
            "level": level,
            "abs_z": r["abs_z"]
        })
    return friendly

# ========= OpenAI Key & 调用 =========
def _get_api_key():
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        pass
    if not key:
        import os
        key = os.getenv("OPENAI_API_KEY")
    return key

def call_llm_explainer(model: str, system_prompt: str, user_prompt: str):
    from openai import OpenAI
    client = OpenAI(api_key=_get_api_key())
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ========= 亲民口吻 Prompt =========
def build_prompts(city: str, decision: dict, top_diffs: list[dict], raw_input: dict | None = None):
    system_prompt = (
        "你是短租平台的用户沟通助手。请用**口语化、易懂、友好**的中文解释原因，"
        "避免专业术语（不要说“z 分数、标准差、阈值、向量距离”等）。"
        "输出结构固定：\n"
        "【一句话结论】用 20 字以内说明大意；\n"
        "【哪里不太对】列 3 点以内，描述“比同类偏高/偏低”，用'明显/有点'等词；\n"
        "【我该怎么做】给 3~5 条操作建议；\n"
        "【温馨提示】如有不确定，提醒可以人工复核。\n"
    )
    human = humanize_diffs(top_diffs)[:5]
    brief = {
        "city": city,
        "decision_type": decision.get("type"),
        "closest_cluster": decision.get("closest_cluster"),
        "distance_to_center": decision.get("distance_to_cluster_center"),
        "cluster_95pct_cutoff": decision.get("abnormal_cutoff"),
        "cluster_size_ratio": decision.get("cluster_size_ratio"),
        "human_diffs": [{"name": d["name"], "direction": d["direction"], "level": d["level"]} for d in human],
        "raw_input_preview": {k: raw_input[k] for k in ["PRICE","BEDROOMS","BEDS","ROOM_TYPE","PROPERTY_TYPE"]
                              if raw_input and (k in raw_input)}
    }
    user_prompt = (
        "请严格按上述结构输出，不要泄露内部算法或门槛名词。\n"
        f"参考摘要（JSON）：\n{json.dumps(brief, ensure_ascii=False, indent=2)}"
    )
    return system_prompt, user_prompt

# ============== UI：全局状态 & 城市选择 ==============
st.title("")

# 会话状态（防止按钮重跑丢失结果 & LLM 缓存）
for k, v in {"last_result": None, "last_user_input": None, "last_city": None, "llm_sig": None, "llm_text": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# 侧边栏
st.sidebar.header("设置")
city_keys = list(CITY_OPTIONS.keys())
city_labels = [CITY_OPTIONS[c]["label"] for c in city_keys]
_city_idx = st.sidebar.selectbox("选择城市 City", list(range(len(city_keys))),
                                 format_func=lambda i: city_labels[i], index=0, key="city_select")
CITY = city_keys[_city_idx]
st.sidebar.caption(f"当前城市：{CITY_OPTIONS[CITY]['label']}")

# 加载该城市工件
try:
    scaler, top_features, X_scaled, cluster_labels = load_artifacts(CITY)
except RuntimeError as e:
    st.error(str(e))
    st.stop()

# Tabs：单条预测 / 批量评估
tab_pred, tab_eval = st.tabs(["🔎 单条预测（亲民版）", "📊 批量评估"])

# ------------------------- 单条预测（亲民版） -------------------------
with tab_pred:
    # 输入卡片
    st.markdown('<div class="kcard glow">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">输入信息</div>', unsafe_allow_html=True)

    with st.form(f"input_form_{CITY}", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            price = st.number_input("PRICE (USD)", min_value=0.0, value=120.0, step=1.0, key="inp_price")
            bedrooms = st.number_input("BEDROOMS", min_value=0.0, value=1.0, step=1.0, key="inp_bedrooms")
            beds = st.number_input("BEDS", min_value=0.0, value=1.0, step=1.0, key="inp_beds")
            reviews_per_month = st.number_input("REVIEWS_PER_MONTH", min_value=0.0, value=0.5, step=0.1, key="inp_rpm")
        with col2:
            availability_365 = st.number_input("AVAILABILITY_365", min_value=0.0, value=120.0, step=1.0, key="inp_avail")
            host_listings_count = st.number_input("HOST_LISTINGS_COUNT", min_value=0.0, value=1.0, step=1.0, key="inp_hlc")

        st.markdown('<div class="section-title" style="margin-top:4px;">类别特征</div>', unsafe_allow_html=True)
        room_type = st.text_input("ROOM_TYPE（如 Entire home/apt）", value="Entire home/apt", key="inp_roomtype")
        property_type = st.text_input("PROPERTY_TYPE（如 Apartment）", value="Apartment", key="inp_proptype")

        default_rare = CITY_OPTIONS[CITY]["default_rare_threshold"]
        rare_threshold = st.slider("Rare 阈值（簇占比）", 0.0, 0.2, float(default_rare), 0.01,
                                   help="小于该簇占比将标记为 rare（若未越过异常距离阈值）",
                                   key="inp_rare_thr")

        submitted = st.form_submit_button("🚀 预测", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 新提交：计算并持久化
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
            "CITY": CITY,
        }
        with st.spinner(f"({CITY}) 推理中…"):
            result = classify_listing_from_raw_input(
                user_input_raw=user_input_raw,
                top_features=top_features,
                scaler=scaler,
                X_scaled=X_scaled,
                cluster_labels=cluster_labels,
                rare_threshold=rare_threshold
            )
        st.session_state["last_result"] = result
        st.session_state["last_user_input"] = user_input_raw
        st.session_state["last_city"] = CITY
        # 新预测后清空上次 LLM 文本，避免错配
        st.session_state["llm_sig"] = None
        st.session_state["llm_text"] = None

    # 展示：会话里有结果就渲染
    has_cached = st.session_state["last_result"] is not None
    if submitted or has_cached:
        result = st.session_state["last_result"] if not submitted else result
        user_input_raw = st.session_state["last_user_input"] if not submitted else user_input_raw
        city_of_result = st.session_state.get("last_city", CITY)

        if city_of_result != CITY:
            st.info(f"当前城市切换为 {CITY}，以下结果来自 {city_of_result}。请点击“🚀 预测”以更新。")

        # —— 亲民结果头部：状态徽章 ——
        label = result.get("type", "unknown")
        badge_html = {
            "typical": '<span class="badge badge-green">✅ 看起来挺正常的</span>',
            "rare": '<span class="badge badge-yellow">🟡 有点少见（不一定是问题）</span>',
            "anomaly": '<span class="badge badge-red">🛑 与同类差距较大，建议自查</span>',
        }.get(label, '<span class="badge">ℹ️ 结果未知</span>')

        st.markdown('<div class="kcard glow" style="margin-top:12px;">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">结果</div>', unsafe_allow_html=True)
        st.markdown(badge_html, unsafe_allow_html=True)

        # —— 位置感知（百分位） & 风险带 ——
        centers, dists_by_label, global_dists = build_distance_index(X_scaled, cluster_labels)
        closest = result.get("closest_cluster")
        user_dist = result.get("distance_to_cluster_center", float("nan"))
        cutoff = result.get("abnormal_cutoff", float("nan"))
        cluster_ratio = result.get("cluster_size_ratio", float("nan"))

        p_in_cluster = percentile_rank(dists_by_label.get(closest, np.array([])), user_dist)
        p_global = percentile_rank(global_dists, user_dist)
        band = size_band_text(cluster_ratio if cluster_ratio == cluster_ratio else 0.0)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**同类中的位置**")
            try:
                import plotly.graph_objects as go
                fig1 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=p_in_cluster,
                    title={"text": "同簇距离百分位（%）"},
                    gauge={"axis":{"range":[0,100]}, "bar":{"thickness":0.25},
                           "steps":[
                               {"range":[0,80], "color":"#17351f"},
                               {"range":[80,95], "color":"#3c3310"},
                               {"range":[95,100], "color":"#3a1518"},
                           ]}
                ))
                st.plotly_chart(fig1, use_container_width=True)
            except Exception:
                st.progress(min(max(int(p_in_cluster),0),100))
                st.caption(f"{p_in_cluster:.1f}%")
            st.caption("数值越高说明与同类差异越大；≥95% 通常视为异常候选。")

        with c2:
            st.markdown("**全城中的位置**")
            try:
                import plotly.graph_objects as go
                fig2 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=p_global,
                    title={"text": "全局距离百分位（%）"},
                    gauge={"axis":{"range":[0,100]}, "bar":{"thickness":0.25},
                           "steps":[
                               {"range":[0,80], "color":"#17351f"},
                               {"range":[80,95], "color":"#3c3310"},
                               {"range":[95,100], "color":"#3a1518"},
                           ]}
                ))
                st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                st.progress(min(max(int(p_global),0),100))
                st.caption(f"{p_global:.1f}%")
            st.caption("与全城所有房源相比的相对位置。")

        st.markdown(f"**该类型的常见度：** {band}（约 {cluster_ratio:.1%} 的房源属于这一类）")

        # —— 同簇距离直方图 + 你的位置 & 95% 阈值 ——
        if closest in dists_by_label:
            try:
                import plotly.express as px
                df_hist = pd.DataFrame({"dist": dists_by_label[closest]})
                figh = px.histogram(df_hist, x="dist", nbins=30, title=f"与你最像的一类（簇 {closest}）的距离分布")
                figh.add_vline(x=user_dist, line_width=3, line_dash="dash", annotation_text="你的位置", annotation_position="top")
                figh.add_vline(x=cutoff, line_width=2, line_dash="dot", line_color="#ff9aa2", annotation_text="95% 阈值", annotation_position="top left")
                st.plotly_chart(figh, use_container_width=True)
            except Exception:
                st.write("（直方图需要 plotly，可选安装：pip install plotly）")

            # ✅ 这里是你提的“解释这张图”的说明（放在图下方）
            st.info(
                "这张图展示的是：在“与你最像的一类房源”里，各个房源到该类中心的距离分布。"
                "靠左代表更接近该类的典型样子，越往右越偏离。虚线是你的位置；"
                "粉色点线是**95% 阈值**（超过它通常视为异常候选）。"
            )

        st.markdown('</div>', unsafe_allow_html=True)  # 结束结果卡片

        # —— 自动生成 LLM 口语化解释（仅 rare/anomaly 时） ——
        if label in {"rare", "anomaly"}:
            # 计算 top_diffs（异常贡献度）作为 LLM 的依据
            mask = (cluster_labels == closest)
            cluster_points = X_scaled[mask]

            df_tmp = pd.DataFrame([user_input_raw])
            df_tmp["LOG_PRICE"] = np.log1p(df_tmp["PRICE"])
            df_tmp["LOG_REVIEWS_PER_MONTH"] = np.log1p(df_tmp["REVIEWS_PER_MONTH"])
            df_tmp["LOG_HOST_LISTINGS_COUNT"] = np.log1p(df_tmp["HOST_LISTINGS_COUNT"])
            df_tmp["LOG_AVAILABILITY_365"] = np.log1p(df_tmp["AVAILABILITY_365"])
            df_tmp["LISTING_DENSITY"] = df_tmp["REVIEWS_PER_MONTH"] / (df_tmp["AVAILABILITY_365"] + 1)
            df_tmp["BEDROOM_BED_RATIO"] = df_tmp["BEDROOMS"] / (df_tmp["BEDS"] + 1)
            room_dum = pd.get_dummies(df_tmp.get("ROOM_TYPE", pd.Series(dtype=str)), prefix="ROOM_TYPE")
            prop_dum = pd.get_dummies(df_tmp.get("PROPERTY_TYPE", pd.Series(dtype=str)), prefix="PROPERTY_TYPE")
            df_tmp = pd.concat([df_tmp, room_dum, prop_dum], axis=1)
            for col in top_features:
                if col not in df_tmp.columns:
                    df_tmp[col] = 0
            X_user_now = df_tmp[top_features]
            X_user_scaled_now = scaler.transform(X_user_now)

            # top_diffs 计算
            mu = cluster_points.mean(axis=0)
            sigma = cluster_points.std(axis=0, ddof=1) + 1e-8
            x = X_user_scaled_now.reshape(-1)
            z = (x - mu) / sigma
            order = np.argsort(-np.abs(z))[:8]
            top_diffs = []
            for idx in order:
                top_diffs.append({
                    "feature": top_features[idx],
                    "z_diff": float(z[idx]),
                    "abs_z": float(abs(z[idx])),
                    "center": float(mu[idx]),
                    "user": float(x[idx])
                })

            # 生成 LLM 签名（避免重复请求）
            sig_src = {
                "city": city_of_result,
                "label": label,
                "closest": int(closest) if closest is not None else -999,
                "user_dist": user_dist,
                "cutoff": cutoff,
                "input_hash": hashlib.md5(json.dumps(user_input_raw, sort_keys=True).encode()).hexdigest(),
            }
            sig = hashlib.md5(json.dumps(sig_src, sort_keys=True).encode()).hexdigest()

            st.markdown('<div class="kcard glow" style="margin-top:12px;">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🧠 口语化解释 & 下一步该怎么做</div>', unsafe_allow_html=True)

            api_key = _get_api_key()
            if not api_key:
                st.warning("缺少 OPENAI_API_KEY（环境变量或 st.secrets）。因此暂不生成 AI 解释。")
            else:
                # 若签名变化或没有缓存，就调用 LLM
                if st.session_state["llm_sig"] != sig or not st.session_state["llm_text"]:
                    system_prompt, user_prompt = build_prompts(
                        city=city_of_result,
                        decision=result,
                        top_diffs=top_diffs,
                        raw_input=user_input_raw
                    )
                    with st.spinner("AI 正在生成口语化解释…"):
                        try:
                            text = call_llm_explainer("gpt-4o-mini", system_prompt, user_prompt)
                            st.session_state["llm_sig"] = sig
                            st.session_state["llm_text"] = text
                        except Exception as e:
                            st.session_state["llm_text"] = None
                            st.error(f"调用 LLM 失败：{e}")

                if st.session_state["llm_text"]:
                    st.chat_message("assistant").markdown(st.session_state["llm_text"])

            st.markdown('</div>', unsafe_allow_html=True)

        # 可选：把百分位等写回 result，便于导出/复用
        result["percentile_in_cluster"] = round(p_in_cluster, 2)
        result["percentile_global"] = round(p_global, 2)
        result["size_band"] = band
        st.session_state["last_result"] = result  # 刷新缓存

# ------------------------- 批量评估 -------------------------
with tab_eval:
    st.markdown('<div class="kcard glow">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">批量评估</div>', unsafe_allow_html=True)

    default_path = Path(__file__).parent / CITY / "full_test_listings.json"
    test_listings = None

    if default_path.exists():
        with open(default_path, "r") as f:
            test_listings = json.load(f)
        st.caption(f"已自动加载：{default_path.as_posix()}")
    else:
        up = st.file_uploader("上传 full_test_listings.json（list[dict]，每条包含 LABEL）", type=["json"], key="eval_uploader")
        if up is not None:
            test_listings = json.load(up)

    if test_listings is not None:
        with st.spinner(f"({CITY}) 评估中…"):
            cm_df, report_df, y_true, y_pred = evaluate_anomaly_detector(
                test_listings=test_listings,
                classify_func=classify_listing_from_raw_input,
                top_features=top_features,
                scaler=scaler,
                X_scaled=X_scaled,
                cluster_labels=cluster_labels
            )

        # 混淆矩阵（有 plotly 则热力图，没装则表格）
        try:
            import plotly.express as px
            fig_cm = px.imshow(cm_df.values,
                               x=list(cm_df.columns), y=list(cm_df.index),
                               text_auto=True, aspect="auto", title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)
        except Exception:
            st.write("### 混淆矩阵")
            st.dataframe(cm_df, use_container_width=True)

        # 分类报告（AgGrid 优先）
        st.write("### 分类报告")
        try:
            from st_aggrid import AgGrid, GridOptionsBuilder
            gob = GridOptionsBuilder.from_dataframe(report_df)
            gob.configure_default_column(resizable=True, sortable=True, filter=True)
            gob.configure_grid_options(domLayout='autoHeight')
            AgGrid(report_df, gridOptions=gob.build(), fit_columns_on_grid_load=True, theme="alpine")
        except Exception:
            st.dataframe(report_df, use_container_width=True)
    else:
        st.info(f"请上传或放置 `{(Path(__file__).parent / CITY / 'full_test_listings.json').as_posix()}` 后再开始评估。")

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------- 页脚 -------------------------
st.markdown('<div class="footer">© AirGuard — anomaly detection & insights</div>', unsafe_allow_html=True)
