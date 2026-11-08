import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
import numpy as np

# ---------------------- 文本预处理函数 ----------------------
def clean_text(text: str) -> str:
    """清洗文本：去除标点、多余空格、换行"""
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------- 文件读取函数 ----------------------
def read_uploaded_file(uploaded_file) -> str:
    if uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif uploaded_file.name.endswith(".docx"):
        doc = Document(uploaded_file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    else:
        return ""

# ---------------------- 余弦相似度批量计算 ----------------------
def calc_similarity_matrix(text_list):
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform(text_list)
    sim_matrix = cosine_similarity(tfidf)
    return sim_matrix

# ---------------------- 页面配置 ----------------------
st.set_page_config(page_title="作业查重与批改助手", layout="wide")
st.title("📄 作业查重与批改助手（余弦相似度）")
st.markdown("本系统使用TF-IDF+余弦相似度算法，计算作业之间重复度，并对照参考答案自动打分")

# 侧边栏上传区
with st.sidebar:
    st.header("📁 文件上传")
    ref_file = st.file_uploader("上传【参考答案】", type=["txt", "docx"])
    student_files = st.file_uploader("上传【学生作业（可多选）】", type=["txt", "docx"], accept_multiple_files=True)
    run_btn = st.button("🚀 开始查重+自动批改", type="primary")

if run_btn:
    if ref_file is None or len(student_files) == 0:
        st.warning("请先上传参考答案 + 至少一份学生作业！")
    else:
        # 读取参考答案
        ref_raw = read_uploaded_file(ref_file)
        ref_clean = clean_text(ref_raw)
        student_names = []
        student_clean_texts = []
        student_raw_texts = []
        for f in student_files:
            student_names.append(f.name)
            raw = read_uploaded_file(f)
            student_raw_texts.append(raw)
            student_clean_texts.append(clean_text(raw))

        all_texts = [ref_clean] + student_clean_texts
        sim_matrix = calc_similarity_matrix(all_texts)

        # 1. 作业两两查重矩阵（学生之间）
        student_sim = sim_matrix[1:, 1:]
        df_sim = pd.DataFrame(student_sim, index=student_names, columns=student_names)
        df_sim = df_sim.round(4)

        # 2. 和参考答案相似度，换算百分制分数
        ref_sims = sim_matrix[1:, 0]
        scores = np.round(ref_sims * 100, 1)
        df_score = pd.DataFrame({
            "文件名": student_names,
            "与参考答案相似度": ref_sims.round(4),
            "自动百分制分数": scores
        })

        st.subheader("✅ 批改结果 - 作业得分表")
        st.dataframe(df_score, use_container_width=True)

        st.subheader("🔍 学生作业两两查重相似度矩阵（重复比）")
        st.dataframe(df_sim, use_container_width=True)

        # 导出csv
        csv_score = df_score.to_csv(index=False, encoding="utf-8-sig")
        csv_sim = df_sim.to_csv(encoding="utf-8-sig")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 下载得分表csv", csv_score, "作业批改分数.csv", mime="text/csv")
        with col2:
            st.download_button("📥 下载查重矩阵csv", csv_sim, "作业相似度矩阵.csv", mime="text/csv")

        # 展示说明
        st.info("💡 说明：相似度越接近1，内容越相似；查重用于发现抄袭，分数仅作为参考，老师仍需人工复核。")
else:
    st.info("👈 左侧上传参考答案和学生作业，点击【开始查重+自动批改】运行")
    with st.expander("📌 算法原理：余弦相似度"):
        st.latex(r"cos(\theta)=\frac{\vec{A}\cdot\vec{B}}{||\vec{A}||\times||\vec{B}||}")
        st.markdown("将文本转为TF-IDF向量，向量夹角余弦值代表文本相似程度，取值区间 [0,1]")
