import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# 设置页面配置，启用宽屏模式
st.set_page_config(layout="wide")

# 页面标题
st.title("Correa Cascade Prediction")

# 加载 LightGBM 模型
try:
    model = joblib.load('./model.m')
except Exception as e:
    st.error(f"Unable to load model, error message：{e}")
    st.stop()

# 定义类别映射
category_mapping = {
    0: "HC",
    1: "NAG",
    2: "AG",
    3: "IM",
    4: "GC"
}

# 创建两个列布局
col1, col2 = st.columns([1, 1])

st.markdown(
    """
    <style>
    /* 设置左侧列背景 */
    div[data-testid="column"]:nth-child(1) {
        background-color: #D9D9D9;
        padding: 20px;
        border-radius: 10px;
    }

    /* 右侧列整体布局，使其与左侧保持对齐 */
    div[data-testid="column"]:nth-child(2) {
        display: flex;
        flex-direction: column;
        justify-content: space-between; /* 保持上下对齐 */
        height: 100%;
    }

    /* 让分页按钮区域紧跟 "Previous Predictions" */
    div[data-testid="column"]:nth-child(2) > div:nth-child(2) {
        padding: 10px;
        display: flex;
        gap: 10px;
        margin-bottom: 0;
        background-color: transparent !important; /* 移除灰色背景 */
    }

    /* 彻底移除 Previous 按钮外层 div 的背景色 */
    div:has(> button[aria-label="Previous"]) {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* 让所有按钮外的 div 透明，防止背景色残留 */
    div[data-testid="stHorizontalBlock"] > div {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* 进一步确保按钮外的 div 没有背景色 */
    div[data-testid="stButton"] {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* 可能的分页容器，确保透明 */
    div[data-testid="column"]:nth-child(2) > div {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* 给数据表格添加边框 */
    div[data-testid="stDataFrame"] {
        border: 2px solid #000 !important; /* 黑色2px边框 */
        border-radius: 5px !important; /* 圆角边框 */
        overflow: hidden !important; /* 防止溢出 */
    }

    /* 让表格的单元格也有边框 */
    div[data-testid="stDataFrame"] table {
        border-collapse: collapse !important; /* 让边框合并 */
        border: 2px solid #000 !important; /* 黑色边框 */
    }

    /* 让每个单元格的边框更明显 */
    div[data-testid="stDataFrame"] td, 
    div[data-testid="stDataFrame"] th {
        border: 1px solid #000 !important; /* 单元格黑色边框 */
        padding: 8px !important; /* 增加填充 */
    }

    """,
    unsafe_allow_html=True
)

# 左侧列：输入表单
with col1:
    st.subheader("New Sample")
    age = st.number_input("Age (0-100)", min_value=0, max_value=100, value=50)
    mono = st.number_input("MONO% (3-10)", min_value=3.0, max_value=10.0, value=5.0)
    ag = st.number_input("A/G (1.2-2.4)", min_value=1.2, max_value=2.4, value=1.5)
    baso = st.number_input("BASO% (0-1)", min_value=0.0, max_value=1.0, value=0.5)
    pdw = st.number_input("PDW (9.9-17)", min_value=9.9, max_value=17.0, value=12.0)
    dbil = st.number_input("DBIL (0-6.8)", min_value=0.0, max_value=6.8, value=5.0)
    neut = st.number_input("NEUT# (1.8-6.3)", min_value=1.8, max_value=6.3, value=5.0)
    lymph = st.number_input("LYMPH# (1.1-3.2)", min_value=1.1, max_value=3.2, value=2.0)
    crea = st.number_input("CREA (53-123)", min_value=53.0, max_value=123.0, value=80.0)
    ast = st.number_input("AST (15-40)", min_value=15.0, max_value=40.0, value=30.0)

    if st.button("Predict", key="predict_button"):  # 添加 key
        input_data = pd.DataFrame({
            'Age': [age],
            'MONO%': [mono],
            'A/G': [ag],
            'BASO%': [baso],
            'PDW': [pdw],
            'DBIL': [dbil],
            'NEUT#': [neut],
            'LYMPH#': [lymph],
            'CREA': [crea],
            'AST': [ast]
        })

        try:
            prediction = model.predict(input_data)
            prediction = int(prediction[0])
            pred_label = category_mapping[prediction]
            new_prediction = {
                'Age': age,
                'MONO%': mono,
                'A/G': ag,
                'BASO%': baso,
                'PDW': pdw,
                'DBIL': dbil,
                'NEUT#': neut,
                'LYMPH#': lymph,
                'CREA': crea,
                'AST': ast,
                'pred': pred_label
            }

            if 'predictions' not in st.session_state:
                st.session_state.predictions = pd.DataFrame()
            st.session_state.predictions = pd.concat([st.session_state.predictions, pd.DataFrame([new_prediction])],
                                                     ignore_index=True)
            if len(st.session_state.predictions) > 10:
                st.session_state.predictions = st.session_state.predictions.tail(10)
        except Exception as e:
            st.error(f"Error message：{e}")

with col2:
    # 在顶部插入图片
    st.image("./Page.jpg", width=800)

    st.subheader("The predict result is:")
    if 'predictions' in st.session_state and not st.session_state.predictions.empty:
        latest_pred = st.session_state.predictions.iloc[-1]
        st.write(f"Prediction: {latest_pred['pred']}")

    st.subheader("Previous Predictions")
    if 'predictions' in st.session_state and not st.session_state.predictions.empty:
        columns_to_show = ['Age', 'MONO%', 'A/G', 'BASO%', 'PDW', 'DBIL', 'NEUT#', 'LYMPH#', 'CREA', 'AST', 'pred']
        st.session_state.predictions_page = st.session_state.get('predictions_page', 1)
        entries_per_page = 5
        total_pages = (len(st.session_state.predictions) + entries_per_page - 1) // entries_per_page

        col_prev, col_page, col_next = st.columns([1, 1, 1])
        with col_prev:
            if st.button("Previous", key="prev_button"):  # 修改为 prev_button
                st.session_state.predictions_page = max(1, st.session_state.predictions_page - 1)
        with col_page:
            st.write(f"Page {st.session_state.predictions_page} of {total_pages}")
        with col_next:
            if st.button("Next", key="next_button"):  # 保持 next_button
                st.session_state.predictions_page = min(total_pages, st.session_state.predictions_page + 1)

        start_idx = (st.session_state.predictions_page - 1) * entries_per_page
        end_idx = start_idx + entries_per_page
        st.dataframe(st.session_state.predictions[columns_to_show].iloc[start_idx:end_idx], height=500, width=800)
    else:
        st.write("No predictions yet.")
