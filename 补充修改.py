import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ================= 修复配置 =================
# 1. 设置字体（解决中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
# 2. 解决负号显示为方框的问题（这一行必须设为 False）
plt.rcParams['axes.unicode_minus'] = False
# ===========================================
st.set_page_config(page_title="HFpEF合并CKD再入院风险预测", layout="wide")

# 变量名映射 (中文显示 -> 英文特征名)
# 顺序基于你的 SHAP 图重要性排序
NAME_MAPPING = {
    "egfr": "eGFR (mL/min/1.73m2)",
    "E_over_e_prime": "E/e' (左室充盈压)",
    "d_dimer": "D-二聚体 (mg/L)",
    "serum_creatinine": "血肌酐 (μmol/L)",
    "nyha_class": "NYHA 心功能分级",
    "serum_uric_acid": "血尿酸 (μmol/L)",
    "blood_urea_nitrogen": "尿素氮 (mmol/L)",
    "nt_probnp": "NT-proBNP (pg/mL)",
    "homocysteine": "同型半胱氨酸 (μmol/L)",
    "hs_crp": "hs-CRP (mg/L)"
}

# 反向映射用于查找
REVERSE_MAPPING = {v: k for k, v in NAME_MAPPING.items()}


# ==========================================
# 2. 加载资源
# ==========================================
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('final_model.pkl')
        background_data = joblib.load('train_data_sample.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, background_data, feature_names
    except FileNotFoundError:
        st.error("⚠️ 未找到模型文件！请确保 final_model.pkl, train_data_sample.pkl, feature_names.pkl 在当前目录下。")
        return None, None, None


model, X_train_bg, model_features = load_resources()

# ==========================================
# 3. 侧边栏：患者数据输入
# ==========================================
st.sidebar.header("🏥 患者临床指标输入")

input_dict = {}

if model_features:
    # 动态生成输入框，确保顺序正确
    # 这里我们手动按照 SHAP 重要性分组展示，体验更好

    st.sidebar.subheader("核心肾脏指标")
    input_dict['egfr'] = st.sidebar.number_input(NAME_MAPPING['egfr'], min_value=5.0, max_value=150.0, value=30.0,
                                                 step=1.0)
    input_dict['serum_creatinine'] = st.sidebar.number_input(NAME_MAPPING['serum_creatinine'], min_value=20.0,
                                                             max_value=1000.0, value=150.0)
    input_dict['blood_urea_nitrogen'] = st.sidebar.number_input(NAME_MAPPING['blood_urea_nitrogen'], min_value=1.0,
                                                                max_value=50.0, value=10.0)

    st.sidebar.subheader("核心心脏指标")
    input_dict['E_over_e_prime'] = st.sidebar.number_input(NAME_MAPPING['E_over_e_prime'], min_value=1.0,
                                                           max_value=50.0, value=15.0)
    input_dict['nt_probnp'] = st.sidebar.number_input(NAME_MAPPING['nt_probnp'], min_value=10.0, max_value=35000.0,
                                                      value=2000.0, step=100.0)
    input_dict['nyha_class'] = st.sidebar.selectbox(NAME_MAPPING['nyha_class'], options=[1, 2, 3, 4], index=2)

    st.sidebar.subheader("生物标志物")
    input_dict['d_dimer'] = st.sidebar.number_input(NAME_MAPPING['d_dimer'], min_value=0.0, max_value=20.0, value=0.5,
                                                    step=0.1)
    input_dict['serum_uric_acid'] = st.sidebar.number_input(NAME_MAPPING['serum_uric_acid'], min_value=50.0,
                                                            max_value=1000.0, value=400.0)
    input_dict['homocysteine'] = st.sidebar.number_input(NAME_MAPPING['homocysteine'], min_value=1.0, max_value=100.0,
                                                         value=15.0)
    input_dict['hs_crp'] = st.sidebar.number_input(NAME_MAPPING['hs_crp'], min_value=0.0, max_value=200.0, value=5.0)

# ==========================================
# 4. 主界面：预测与解释
# ==========================================
st.title("❤️ HFpEF合并CKD再入院风险智能评估系统")
st.markdown("基于 Logistic Regression 与 SHAP 可解释性算法")

if st.button("🚀 开始评估", type="primary"):
    if model is None:
        st.stop()

    # 1. 构建输入 DataFrame (确保列顺序与训练时一致)
    input_df = pd.DataFrame([input_dict])
    # 确保只包含模型需要的列，且顺序一致
    input_df = input_df[model_features]

    # 2. 模型预测
    # 注意：你的模型可能是 CalibratedClassifierCV，需要用 predict_proba
    try:
        prob = model.predict_proba(input_df)[:, 1][0]
    except:
        st.error("模型结构异常，无法调用 predict_proba")
        st.stop()

    # 3. 显示结果
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("预测结果")
        risk_percentage = prob * 100

        # 动态颜色
        if risk_percentage < 30:
            color = "green"
            level = "低风险"
        elif risk_percentage < 70:
            color = "orange"
            level = "中风险 (灰色地带)"
        else:
            color = "red"
            level = "高风险"

        st.markdown(f"""
        <div style="text-align: center; border: 2px solid {color}; padding: 20px; border-radius: 10px;">
            <h1 style="color: {color}; font-size: 50px;">{risk_percentage:.1f}%</h1>
            <h3>{level}</h3>
        </div>
        """, unsafe_allow_html=True)
        st.info("注：该概率指患者在出院后1年内发生因心衰再入院的可能性。")

    with col2:
        st.subheader("风险归因分析 (SHAP)")
        with st.spinner("正在计算特征贡献度..."):
            # SHAP 计算逻辑
            # 需要从 CalibratedClassifierCV 或 Pipeline 中提取核心 LR 模型
            estimator = model
            if hasattr(estimator, 'calibrated_classifiers_'):
                estimator = estimator.calibrated_classifiers_[0].estimator
            if hasattr(estimator, 'named_steps'):
                # 如果是 Pipeline，我们需要取出 Step 里的 Model
                # 并且我们需要先对数据进行 Pipeline 前半部分的预处理 (Scaler)
                scaler = estimator.named_steps['scaler']
                clf = estimator.named_steps['clf']

                # 预处理背景数据和输入数据
                X_bg_scaled = scaler.transform(X_train_bg)
                X_input_scaled = scaler.transform(input_df)

                # 创建解释器 (针对 LR 后的线性部分)
                explainer = shap.LinearExplainer(clf, X_bg_scaled, feature_perturbation="interventional")
                shap_values = explainer(X_input_scaled)

            else:
                # 如果没有 Pipeline 直接是模型
                explainer = shap.LinearExplainer(estimator, X_train_bg, feature_perturbation="interventional")
                shap_values = explainer(input_df)

            # 修正 SHAP 对象的 feature_names 为中文，方便展示
            shap_values.feature_names = [NAME_MAPPING.get(c, c) for c in model_features]

            # 绘制瀑布图
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(fig)

    # 文字解释
    st.markdown("---")
    st.subheader("🤖 AI 分析报告")
    top_feature_idx = np.argmax(np.abs(shap_values.values[0]))
    top_feature_name = shap_values.feature_names[top_feature_idx]
    contribution = shap_values.values[0][top_feature_idx]
    direction = "增加" if contribution > 0 else "降低"

    st.write(f"根据模型分析，对该患者风险影响最大的因素是 **{top_feature_name}**，"
             f"它使再入院概率**{direction}**了 **{abs(contribution) * 100:.1f}%**。")

else:
    st.info("👈 请在左侧侧边栏输入患者指标，然后点击“开始评估”。")